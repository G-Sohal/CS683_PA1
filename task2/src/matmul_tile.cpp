// matmul_tile.cpp — STAGE 4: CACHE TILING (BLOCKING)   [TODO]
//
// Your Stage-3 SIMD kernel is now fast enough that, at large sizes, it is limited by MEMORY,
// not arithmetic: it re-streams B (and A) from DRAM far more often than the caches can hold.
// (This is why tiling did nothing for the scalar stages — those were compute-bound — but pays
// off now.)
//
// Fix it with cache blocking: split M and N (and optionally K) into blocks (Mc, Nc) chosen so
// the active A panel (Mc x K) and B panel (Nc x K) fit in L1/L2. Loop over blocks on the
// outside; run your Stage-3 register-tiled micro-kernel on the inside. Each panel is then
// pulled into cache once per block and reused by every micro-tile in it, instead of being
// refetched from DRAM on every pass.
//
// Why it matters: matmul has O(N^3) work over O(N^2) data — enormous reuse. Keeping the active
// panels in cache is what turns the memory-bound SIMD kernel back into a compute-bound one.
// The effect grows with size: negligible when the matrices fit in cache, large at 2048+ (try
// `./bin/matmul all 2048 2048 2048` and compare your simd vs tile rows). This is the tiling
// payoff matrix multiply has and convolution (Task 1) did not.
//
// Hint: two outer block loops (over N, then M) wrapping your micro-kernel; start near
// Mc=Nc=128 and sweep.

#include <immintrin.h>

#include "matmul.h"

void matmul_tile(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO: replace this placeholder with your cache-blocked SIMD implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
