// matmul_optimized.cpp — STAGE 6: PUT IT ALL TOGETHER   [TODO]
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything — register-tiled AVX2/FMA micro-kernel, cache blocking, and software prefetch
// — and TUNE it to be as fast as you can. Your speedup over matmul_naive determines your
// score (see the tier table the harness prints), and this same function will power a real
// LLM inference via `make llama-demo`.
//
// Worth trying:
//   - A well-sized register micro-kernel (e.g. 8x4, 8x6, 16x2 outputs) that maximizes FMA
//     issue while fitting in the 16 AVX registers.
//   - Cache block sizes (Mc, Nc, Kc) tuned to your L1/L2 (check `lscpu`).
//   - Packing the A and B panels into small contiguous buffers so the micro-kernel reads
//     them with unit stride and no TLB/stride surprises.
//   - Prefetch distance tuning.
//   - __restrict__ pointers.
//
// It must stay correct for ARBITRARY lda/ldb/ldc (not just the contiguous microbenchmark
// case), because llama.cpp will call it with strides from its tensors. Measure after every
// change — let the numbers decide.

#include <immintrin.h>

#include "matmul.h"

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO: replace this placeholder with your best combined implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
