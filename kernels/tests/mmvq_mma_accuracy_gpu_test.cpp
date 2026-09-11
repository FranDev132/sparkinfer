// Dumps a Q4_K attention projection at Muse Glimmer's real shapes so the tensor-core path and the
// per-row MMVQ can be diffed across processes:
//
//     SPARKINFER_MMVQ_MMA=0 MMVQ_MMA_DUMP=/tmp/mmvq.bin ./mmvq_mma_accuracy_gpu_test 32 6656 4096
//     SPARKINFER_MMVQ_MMA=1 MMVQ_MMA_DUMP=/tmp/mma.bin  ./mmvq_mma_accuracy_gpu_test 32 6656 4096
//
// Two processes, not two calls: launch_mmvq_rows caches its env read in a function-local static.
//
// Deliberately NOT a bit-identity test. The mma path reduces over K in a different order (one
// m16n8k32 per 32-value scale group, accumulated in float) than the per-row dp4a MMVQ, so they
// cannot agree bit-for-bit. What must hold is that the gap is rounding rather than a different
// answer. Unlike the FFN down projection, these outputs feed the attention computation rather than
// the residual, so the agreement is worth measuring here in its own right.
#include "sparkinfer/kernels/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {
struct blk_q8_1 { __half2 ds; signed char qs[32]; };
struct blk_q4_K { __half2 dm; unsigned char scales[12]; unsigned char qs[128]; };
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
    const int M = argc > 1 ? atoi(argv[1]) : 32;     // packed batch width
    const int N = argc > 2 ? atoi(argv[2]) : 6656;   // output dim (o-proj)
    const int K = argc > 3 ? atoi(argv[3]) : 4096;   // input dim
    const int nblk = K / 256, nq8 = K / 32;

    std::vector<blk_q4_K> hW((size_t)N * nblk);
    uint32_t seed = 0x5EEDu;                          // fixed: both processes build the same inputs
    for (auto& b : hW) {
        b.dm = __floats2half2_rn(0.008f + (lcg(seed) % 64) * 1e-4f,
                                 0.004f + (lcg(seed) % 32) * 1e-4f);
        for (int i = 0; i < 12; i++) b.scales[i] = (unsigned char)(lcg(seed) & 0x3f);
        for (int i = 0; i < 128; i++) b.qs[i] = (unsigned char)(lcg(seed) & 0xff);
    }
    std::vector<blk_q8_1> hA((size_t)M * nq8);
    for (auto& b : hA) {
        const float d = 0.002f + (lcg(seed) % 31) * 1e-4f;
        int s = 0;
        for (int i = 0; i < 32; i++) { b.qs[i] = (signed char)((int)(lcg(seed) & 0xff) - 128); s += b.qs[i]; }
        b.ds = __floats2half2_rn(d, d * (float)s);
    }

    auto* dW = dev<blk_q4_K>(hW.size(), hW.data());
    auto* dA = dev<blk_q8_1>(hA.size(), hA.data());
    auto* dY = dev<__nv_bfloat16>((size_t)M * N);
    if (!dW || !dA || !dY) { std::printf("[SKIP] allocation failed\n"); return 77; }

    if (!sparkinfer::kernels::launch_mmvq_rows(12, dA, dW, dY, M, N, K, nullptr)) {
        std::printf("[FAIL] launch_mmvq_rows declined M=%d N=%d K=%d\n", M, N, K);
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("[FAIL] %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::vector<__nv_bfloat16> out((size_t)M * N);
    cudaMemcpy(out.data(), dY, out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (auto v : out) { const double x = (double)__bfloat162float(v); sum += x * x; }
    const char* mode = getenv("SPARKINFER_MMVQ_MMA");
    std::printf("M=%d N=%d K=%d mode=%s rms=%.9e n=%zu\n",
                M, N, K, mode ? mode : "(default)", std::sqrt(sum / (double)out.size()), out.size());
    if (const char* path = getenv("MMVQ_MMA_DUMP")) {
        FILE* f = fopen(path, "wb");
        if (!f) { std::printf("[FAIL] cannot write %s\n", path); return 1; }
        fwrite(out.data(), sizeof(__nv_bfloat16), out.size(), f);
        fclose(f);
        std::printf("wrote %s\n", path);
    }
    cudaFree(dW); cudaFree(dA); cudaFree(dY);
    return 0;
}
