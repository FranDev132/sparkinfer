// The packed LM head on the tensor cores against the kernel it replaces, at Muse Glimmer's real
// 202048 x 6656 head.
//
// The reference is launch_mmvq_rows_f32, NOT launch_gemv_q4k_dp4a_multirow_f32: the latter is
// instantiated for K in {2048, 5120, 6144} only, so at Muse's K=6656 it declines and the
// continuous-batch head has always fallen through to launch_mmvq_rows_f32 -- which chunks at eight
// rows and runs si_mmvq_q4k_rows_exact_kernel. (The comment at that call site names the multi-row
// kernel, which is misleading for this model.)
//
// The head is scored to be ARGMAXED, so agreement in the argmax is the property that matters and
// the one this asserts. The two kernels reduce over K in a different order, so the logits
// themselves cannot agree bit-for-bit -- the dp4a multi-row kernel is already not bit-identical to
// the rows kernel AR scores with, which is why both are restricted to the packed path. What must
// hold is that a different reduction order does not move the arg of the maximum.
//
// Both arms run in ONE process: they are two exported functions, not two branches behind a cached
// env read, so no dump/diff across processes is needed here.
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/qtype.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
constexpr int VOCAB = 202048, H = 6656;
uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
}  // namespace

int main(int argc, char** argv) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) return 77;
    const int M = argc > 1 ? atoi(argv[1]) : 32;

    const size_t w_bytes = (size_t)VOCAB * (H / 256) * 144;      // ~756 MB
    std::vector<unsigned char> hw(w_bytes);
    uint32_t seed = 0x4EAD1234u;
    for (size_t i = 0; i < w_bytes; ++i) hw[i] = (unsigned char)(lcg(seed) >> 17);
    // A random dm half2 makes the dequantised weights overflow and the comparison meaningless, so
    // write a realistic scale pair into every super-block and keep the 6-bit fields in range.
    for (size_t o = 0; o < w_bytes; o += 144) {
        const __half2 dm = __floats2half2_rn(0.008f + (lcg(seed) % 64) * 1e-4f,
                                             0.004f + (lcg(seed) % 32) * 1e-4f);
        std::memcpy(&hw[o], &dm, sizeof(__half2));
        for (int i = 0; i < 12; i++) hw[o + 4 + i] = (unsigned char)(lcg(seed) & 0x3f);
    }

    // Q8_1 is { __half2 ds; signed char qs[32] } per 32 values. Random bytes over the whole block
    // would randomise ds too, and a random half is very often inf or NaN -- which makes every
    // logit NaN and an argmax comparison meaningless (it would "agree" on index 0 in both arms).
    // Write the quantised values randomly and the scale pair the way the quantiser would:
    // ds.x = d, ds.y = d * sum(q).
    const size_t q_row = sparkinfer::kernels::llama_q8_1_bytes(H);
    std::vector<unsigned char> hq((size_t)M * q_row);
    for (size_t off = 0; off + 36 <= hq.size(); off += 36) {
        int sum = 0;
        for (int i = 0; i < 32; i++) {
            const signed char q = (signed char)(int)((lcg(seed) >> 20) % 255 - 127);
            hq[off + 4 + i] = (unsigned char)q;
            sum += q;
        }
        const float d = 0.005f + (lcg(seed) % 32) * 1e-4f;
        const __half2 ds = __floats2half2_rn(d, d * (float)sum);
        std::memcpy(&hq[off], &ds, sizeof(__half2));
    }

    unsigned char *dw = nullptr, *dq = nullptr;
    float *y_mma = nullptr, *y_dp4a = nullptr;
    if (cudaMalloc(&dw, w_bytes) != cudaSuccess ||
        cudaMalloc(&dq, hq.size()) != cudaSuccess ||
        cudaMalloc(&y_mma, (size_t)M * VOCAB * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&y_dp4a, (size_t)M * VOCAB * sizeof(float)) != cudaSuccess) {
        std::printf("[SKIP] allocation failed (needs ~%.1f GB)\n",
                    (double)(w_bytes + 2.0 * M * VOCAB * 4) / 1e9);
        return 77;
    }
    cudaMemcpy(dw, hw.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice);

    if (!sparkinfer::kernels::launch_mmvq_q4k_mma_head_f32(dq, dw, y_mma, M, VOCAB, H, nullptr)) {
        std::printf("[FAIL] mma head declined M=%d\n", M); return 1;
    }
    // Whole batch in one call: launch_mmvq_rows_f32 does its own eight-row chunking, which is
    // exactly the behaviour being replaced.
    if (!sparkinfer::kernels::launch_mmvq_rows_f32(12, dq, dw, y_dp4a, M, VOCAB, H, nullptr)) {
        std::printf("[FAIL] MMVQ reference declined M=%d N=%d K=%d\n", M, VOCAB, H); return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("[FAIL] %s\n", cudaGetErrorString(cudaGetLastError())); return 1;
    }

    std::vector<float> a((size_t)M * VOCAB), b((size_t)M * VOCAB);
    cudaMemcpy(a.data(), y_dp4a, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), y_mma,  b.size() * sizeof(float), cudaMemcpyDeviceToHost);

    int agree = 0;
    double sum = 0.0, dsum = 0.0, dmax = 0.0;
    for (int r = 0; r < M; ++r) {
        int ia = 0, ib = 0;
        for (int v = 1; v < VOCAB; ++v) {
            if (a[(size_t)r * VOCAB + v] > a[(size_t)r * VOCAB + ia]) ia = v;
            if (b[(size_t)r * VOCAB + v] > b[(size_t)r * VOCAB + ib]) ib = v;
        }
        if (ia == ib) agree++;
        else std::printf("  row %d: dp4a argmax %d (%.6f) vs mma %d (%.6f)\n",
                         r, ia, a[(size_t)r * VOCAB + ia], ib, b[(size_t)r * VOCAB + ib]);
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        sum += (double)a[i] * (double)a[i]; dsum += d * d;
        if (std::fabs(d) > dmax) dmax = std::fabs(d);
    }
    const double rms = std::sqrt(sum / (double)a.size());
    std::printf("M=%d argmax agree %d/%d  relRMS %.4e  worst/rms %.4e\n",
                M, agree, M, std::sqrt(dsum / (double)a.size()) / rms, dmax / rms);

    cudaFree(dw); cudaFree(dq); cudaFree(y_mma); cudaFree(y_dp4a);
    if (!std::isfinite(rms) || rms == 0.0) {
        std::printf("[FAIL] reference logits are not finite (rms=%g) -- the inputs are degenerate "
                    "and an argmax comparison would be meaningless\n", rms);
        return 1;
    }
    if (agree != M) { std::printf("[FAIL] argmax disagrees on %d of %d rows\n", M - agree, M); return 1; }
    std::printf("[PASS] head mma: argmax identical on all %d rows\n", M);
    return 0;
}
