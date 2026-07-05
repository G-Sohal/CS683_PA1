// conv_reorder.cpp — STAGE 1: LOOP REORDERING   [TODO: implement]
//
// Goal: reorder the loop nest so the innermost loop is a long, unit-stride stream
// over the output columns.
//
// Idea: pull the two kernel loops (ky, kx) to the OUTSIDE. For a fixed kernel tap,
// its weight ker[ky*K+kx] is a single scalar, and the work becomes:
//
//     out[y][x] += ker[ky][kx] * in[y+ky][x+kx]     // for all (y, x)
//
// i.e. an AXPY (scaled vector add) sweeping the whole output with unit stride in x.
// Remember to zero `out` first, since you now accumulate into it across taps.
//
// Why it matters: the inner loop now touches contiguous memory that the hardware
// prefetcher loves, instead of the strided gather the naive kernel does — and it sets
// up the clean vectorizable shape that later stages exploit.
//
// Heads up: this "obvious" reorder sweeps the ENTIRE output image once per kernel tap
// (K*K passes over the whole array), which does NOT fit in cache. So at K=3 it may run
// about the same as — or even slightly slower than — naive. That is expected: you have
// fixed the access *pattern* but not yet the *locality*. Stage 3 (tiling) fixes it.
//
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO: replace this placeholder with your reordered implementation.
    conv_naive(in, out, ker, H, W, K);
}
