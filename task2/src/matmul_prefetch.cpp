// matmul_prefetch.cpp — STAGE 5: SOFTWARE PREFETCHING   [TODO]  (NEW technique)
//
// Build on your tiled + SIMD kernel and add software prefetch hints so data arrives in
// cache BEFORE the kernel needs it, hiding memory latency behind the FMAs already running.
//
// Use the intrinsic:
//     _mm_prefetch((const char*)addr, _MM_HINT_T0);   // fetch into L1 (T0), or T1/T2 for L2/L3
//
// Prefetch a few iterations / one panel AHEAD of the current work — e.g. while multiplying
// the current Kc-block or the current B row, prefetch the next A panel row or the next B
// row you will touch. The right PREFETCH DISTANCE (how far ahead) is a tuning knob: too
// close does nothing, too far evicts useful data. Measure several distances.
//
// Why it matters: even with cache blocking, moving to the next panel/row incurs cache
// misses that stall the pipeline; a well-placed prefetch overlaps that fetch with compute.
//
// Reality check: modern cores have strong hardware prefetchers that already handle regular
// streaming, so the gain here is often modest — the skill is knowing WHEN and HOW FAR to
// prefetch, and honestly measuring whether it helped.

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO: replace this placeholder with your tiled+SIMD+prefetch implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
