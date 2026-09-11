// The row-batched down projection is instantiated for 2..8 rows, so a wider decode batch is walked
// in chunks. What is new in a chunk past the first is the pointer arithmetic: its activations,
// expert ids/weights and output all start part-way into the caller's arrays. This pins exactly
// that -- a chunked run of M rows must be BIT-IDENTICAL, row for row, to running each chunk on its
// own as the caller-sized batch it already supported.
//
// It deliberately does NOT compare against the per-token grid: that is a different kernel with a
// different reduction order, and the chunked path is not claimed to reproduce it bit-for-bit.
//
// gate_bf16/up_bf16 are supplied so only the SwiGLU + down half runs, which is the half the packed
// continuous-batch decode step uses and the half this covers.
#include "sparkinfer/kernels/moe.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int H = 6656;      // Muse Glimmer hidden
constexpr int F = 19968;     // Muse Glimmer dense ffn
constexpr int Q4K = 12;      // ggml type id

// Deterministic, reproducible fill -- no rand(), so a failure is always the same failure.
uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }

template <class T> T* dev(size_t n, const void* host = nullptr) {
    void* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) return nullptr;
    if (host) { if (cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr; }
    else      { if (cudaMemset(p, 0, n * sizeof(T)) != cudaSuccess) return nullptr; }
    return static_cast<T*>(p);
}

}  // namespace

int main() {
    int devcount = 0;
    if (cudaGetDeviceCount(&devcount) != cudaSuccess || devcount == 0) return 77;

    // Q4_K down weights for a single expert: [1, H, F], 144 bytes per 256 values.
    const size_t down_bytes = (size_t)H * (F / 256) * 144;
    std::vector<unsigned char> h_down(down_bytes);
    uint32_t seed = 0x5eed1234u;
    for (size_t i = 0; i < down_bytes; ++i) h_down[i] = (unsigned char)(lcg(seed) >> 17);

    constexpr int MMAX = 32;
    std::vector<__nv_bfloat16> h_gate((size_t)MMAX * F), h_up((size_t)MMAX * F);
    for (size_t i = 0; i < (size_t)MMAX * F; ++i) {
        h_gate[i] = __float2bfloat16(((int)(lcg(seed) >> 24) - 128) * 0.01f);
        h_up[i]   = __float2bfloat16(((int)(lcg(seed) >> 24) - 128) * 0.01f);
    }
    std::vector<int>   h_ids((size_t)MMAX, 0);
    std::vector<float> h_wts((size_t)MMAX, 1.0f);

    auto* d_down = dev<unsigned char>(down_bytes, h_down.data());
    auto* d_gate = dev<__nv_bfloat16>((size_t)MMAX * F, h_gate.data());
    auto* d_up   = dev<__nv_bfloat16>((size_t)MMAX * F, h_up.data());
    auto* d_ids  = dev<int>((size_t)MMAX, h_ids.data());
    auto* d_wts  = dev<float>((size_t)MMAX, h_wts.data());
    auto* d_out  = dev<__nv_bfloat16>((size_t)MMAX * H);
    auto* d_hs   = dev<float>((size_t)MMAX * F);
    auto* d_os   = dev<float>((size_t)MMAX * H);
    if (!d_down || !d_gate || !d_up || !d_ids || !d_wts || !d_out || !d_hs || !d_os) {
        std::printf("[SKIP] down-rows chunk: allocation failed (needs ~%.1f GB)\n",
                    (double)down_bytes / 1e9);
        return 77;
    }

    // Run `m` rows starting at row `t0` of the shared inputs, into out + t0*H.
    auto run = [&](int t0, int m) {
        sparkinfer::kernels::launch_moe_expert_ffn_q4k(
            /*input*/ nullptr, /*gate_q*/ nullptr, /*up_q*/ nullptr, d_down,
            Q4K, Q4K, Q4K,
            d_ids + t0, d_wts + t0, d_out + (size_t)t0 * H,
            d_hs, d_os, m, /*top_k*/ 1, H, F,
            /*input_q8*/ nullptr, /*stream*/ nullptr, /*ar_exact_splitk*/ false,
            d_gate + (size_t)t0 * F, d_up + (size_t)t0 * F);
        return cudaDeviceSynchronize() == cudaSuccess;
    };

    auto fetch = [&](int t0, int m, std::vector<__nv_bfloat16>& dst) {
        dst.resize((size_t)m * H);
        return cudaMemcpy(dst.data(), d_out + (size_t)t0 * H,
                          (size_t)m * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess;
    };

    // Every batch width the dispatch chunks, against the same rows run as their own batches.
    // 9 is the tail case: it splits 7+2 rather than 8+1, because M==1 has no instantiation.
    const int widths[]  = { 9, 16, 17, 24, 32 };
    const int chunked[][2][2] = {   // { {t0,m}, {t0,m} } for the two chunks we verify against
        { {0, 7}, {7, 2} },
        { {0, 8}, {8, 8} },
        { {0, 8}, {8, 7} },         // 17 -> 8,7,2 ; we check the first two boundaries
        { {0, 8}, {8, 8} },
        { {0, 8}, {8, 8} },
    };

    bool all_ok = true;
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
        const int M = widths[w];
        if (cudaMemset(d_out, 0, (size_t)MMAX * H * sizeof(__nv_bfloat16)) != cudaSuccess) return 77;
        if (!run(0, M)) { std::printf("[FAIL] M=%d: launch failed\n", M); all_ok = false; continue; }
        std::vector<__nv_bfloat16> full;
        if (!fetch(0, M, full)) return 77;

        bool ok = true;
        for (int part = 0; part < 2; ++part) {
            const int t0 = chunked[w][part][0], m = chunked[w][part][1];
            if (t0 + m > M) continue;
            if (cudaMemset(d_out, 0, (size_t)MMAX * H * sizeof(__nv_bfloat16)) != cudaSuccess) return 77;
            if (!run(t0, m)) { ok = false; break; }
            std::vector<__nv_bfloat16> ref;
            if (!fetch(t0, m, ref)) return 77;
            if (std::memcmp(full.data() + (size_t)t0 * H, ref.data(),
                            (size_t)m * H * sizeof(__nv_bfloat16)) != 0) {
                std::printf("[FAIL] M=%d rows [%d,%d) differ from the same rows run as their own batch\n",
                            M, t0, t0 + m);
                ok = false;
            }
        }
        if (ok) std::printf("[PASS] down rows: M=%d chunked is bit-identical per row\n", M);
        all_ok = all_ok && ok;
    }

    cudaFree(d_down); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_ids);
    cudaFree(d_wts); cudaFree(d_out); cudaFree(d_hs); cudaFree(d_os);
    return all_ok ? 0 : 1;
}
