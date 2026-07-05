// conv_optimized.cpp — STAGE 5: PUT IT ALL TOGETHER   [TODO: implement]
//
// This is the graded function. Combine everything — loop reordering, unrolling,
// cache tiling, and AVX2 — and then TUNE it to be as fast as you can. Your speedup
// over conv_naive on the grading workload determines your Stage-5 score (see the
// tier table printed by the harness / the rubric in README.md).
//
// Things worth trying beyond a naive combination:
//   - Sweep tile sizes for your machine's L1/L2 (print `lscpu` / check the cache sizes).
//   - Unroll the column loop by 2 AVX2 vectors (16 columns) for more ILP.
//   - Use aligned loads/stores where the layout permits, and __restrict__ pointers.
//   - Keep several output-row accumulators live to reuse broadcasted weights.
//   - Consider software prefetching for the next tile.
//
// Correctness is still required: your result must match conv_naive within tolerance,
// or this stage scores 0 regardless of speed.
//
// Hint: measure after every change. Not every "optimization" helps — let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO: replace this placeholder with your best combined implementation.
    conv_naive(in, out, ker, H, W, K);
}
