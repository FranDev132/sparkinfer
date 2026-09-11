// Dumps the dense down-projection output at Muse Glimmer's real 6656x19968 shape so the two
// implementations can be diffed across processes:
//
//     SPARKINFER_DOWN_MMA=0 DOWN_MMA_DUMP=/tmp/mmvq.bin ./down_mma_accuracy_gpu_test
//     SPARKINFER_DOWN_MMA=1 DOWN_MMA_DUMP=/tmp/mma.bin  ./down_mma_accuracy_gpu_test
//
// Two processes, not two calls: the dispatch caches its env read in a function-local static, so a
// single process can only ever exercise one arm.
//
// This is deliberately NOT a bit-identity test. The mma path reduces over K in a different order
// (one m16n8k32 per 32-value scale group, accumulated in float) than the per-row dp4a MMVQ, so the
// two cannot agree bit-for-bit. What must hold is that the gap is rounding rather than a different
// answer, which is what comparing the two dumps shows.
#include "sparkinfer/kernels/moe.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_fp16.h>

namespace {
constexpr int H = 6656, F = 19968, Q4K = 12;
uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
template <class T> T* dev(size_t n, const void* host = nullptr) {
    void* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) return nullptr;
    if (host) cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice);
    else      cudaMemset(p, 0, n * sizeof(T));
    return static_cast<T*>(p);
}
}  // namespace

int main(int argc, char** argv) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) return 77;
    const int M = argc > 1 ? atoi(argv[1]) : 32;

    const size_t down_bytes = (size_t)H * (F / 256) * 144;
    std::vector<unsigned char> h_down(down_bytes);
    uint32_t seed = 0xA11CE5u;                       // fixed: both processes build identical inputs
    for (size_t i = 0; i < down_bytes; ++i) h_down[i] = (unsigned char)(lcg(seed) >> 17);
    // Random bytes across the whole block would randomise the dm/dmin half2 too, which makes the
    // dequantised weights overflow and the comparison meaningless (rms comes out NaN). Write a
    // realistic scale pair into every super-block, and keep the 6-bit scale fields in range.
    for (size_t b = 0; b < down_bytes; b += 144) {
        const float d    = 0.008f + (lcg(seed) % 64) * 1e-4f;
        const float dmin = 0.004f + (lcg(seed) % 32) * 1e-4f;
        const __half2 dm = __floats2half2_rn(d, dmin);
        std::memcpy(&h_down[b], &dm, sizeof(__half2));
        for (int i = 0; i < 12; i++) h_down[b + 4 + i] = (unsigned char)(lcg(seed) & 0x3f);
    }
    std::vector<__nv_bfloat16> h_gu((size_t)M * F);
    for (auto& v : h_gu) v = __float2bfloat16(((int)(lcg(seed) >> 24) - 128) * 0.01f);
    std::vector<int>   h_ids((size_t)M, 0);
    std::vector<float> h_wts((size_t)M, 1.0f);

    auto* d_down = dev<unsigned char>(down_bytes, h_down.data());
    auto* d_gate = dev<__nv_bfloat16>((size_t)M * F, h_gu.data());
    auto* d_up   = dev<__nv_bfloat16>((size_t)M * F, h_gu.data());
    auto* d_ids  = dev<int>((size_t)M, h_ids.data());
    auto* d_wts  = dev<float>((size_t)M, h_wts.data());
    auto* d_out  = dev<__nv_bfloat16>((size_t)M * H);
    auto* d_hs   = dev<float>((size_t)M * F);
    auto* d_os   = dev<float>((size_t)M * H);
    if (!d_down || !d_gate || !d_up || !d_ids || !d_wts || !d_out || !d_hs || !d_os) {
        std::printf("[SKIP] allocation failed (needs ~%.1f GB)\n", (double)down_bytes / 1e9);
        return 77;
    }

    sparkinfer::kernels::launch_moe_expert_ffn_q4k(
        nullptr, nullptr, nullptr, d_down, Q4K, Q4K, Q4K,
        d_ids, d_wts, d_out, d_hs, d_os, M, 1, H, F,
        nullptr, nullptr, false, d_gate, d_up);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("[FAIL] launch: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::vector<__nv_bfloat16> out((size_t)M * H);
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    double sum = 0.0, amax = 0.0;
    for (size_t i = 0; i < out.size(); i++) {
        const double v = (double)__bfloat162float(out[i]);
        sum += v * v; if (std::fabs(v) > amax) amax = std::fabs(v);
    }
    const char* mode = getenv("SPARKINFER_DOWN_MMA");
    std::printf("M=%d mode=%s rms=%.9e amax=%.9e n=%zu\n",
                M, mode ? mode : "(default)", std::sqrt(sum / (double)out.size()), amax, out.size());
    if (const char* path = getenv("DOWN_MMA_DUMP")) {
        FILE* f = fopen(path, "wb");
        if (!f) { std::printf("[FAIL] cannot write %s\n", path); return 1; }
        fwrite(out.data(), sizeof(__nv_bfloat16), out.size(), f);
        fclose(f);
        std::printf("wrote %s\n", path);
    }

    cudaFree(d_down); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_ids);
    cudaFree(d_wts); cudaFree(d_out); cudaFree(d_hs); cudaFree(d_os);
    return 0;
}
