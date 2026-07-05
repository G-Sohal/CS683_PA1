// matmul_simd.cpp — STAGE 3: SIMD with AVX2 intrinsics   [TODO]
//
// Vectorize the K dot product 8 floats at a time with AVX2 + FMA. Because K is the
// contiguous inner dimension of BOTH A and B, the dot product vectorizes directly:
//
//     __m256 acc = _mm256_setzero_ps();
//     for (p = 0; p < K; p += 8)
//         acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p), acc);
//     // then horizontally reduce `acc` to one float
//
// Do NOT stop at a single dot product: keep a register tile of accumulators (e.g. an
// Mr x Nr block of C, Mr*Nr independent __m256 accumulators) so each loaded A vector is
// reused across Nr B rows and each B vector across Mr A rows. Reduce each accumulator once
// at the end of the K loop. A 4x2 tile (8 accumulators) is a solid starting point.
//
// Do the plain, un-blocked sweep here (cache blocking is the NEXT stage). Once this kernel
// runs this fast, it becomes memory-bound at large sizes — which is exactly what Stage 4 fixes.
//
// Why it matters: the vector unit does 8 (AVX2) FMAs per instruction; scalar code wastes
// 7/8 of it. Register tiling raises compute-per-load so you approach the FMA peak.
//
// You need:  #include <immintrin.h>   (AVX2 + FMA are enabled by the Makefile flags.)
// For horizontal reduction of a __m256: add the high and low 128-bit halves, then reduce.

#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO: replace this placeholder with your register-tiled AVX2 implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
