// conv_tile.cpp — STAGE 3: CACHE TILING (BLOCKING)   [TODO: implement]
//
// This is the payoff stage. Your reordered kernel makes K*K separate passes over the
// ENTIRE output image (one pass per kernel tap). For a 2048x2048 float image that is
// ~16 MB streamed K*K times — far larger than L2, so each pass refetches from memory
// and Stages 1-2 barely beat (or even lose to) naive.
//
// Fix it by tiling: split the output into blocks that fit in cache — a strip of TH rows
// (TH*W floats) is the simplest choice — and perform ALL K*K taps on a block before
// moving to the next one. The block's output then stays resident in L1/L2 across every
// tap, turning repeated DRAM traffic into cache hits. Expect a clear jump here (and an
// even bigger one for larger K, where there are more taps to amortize).
//
// Why it matters: for a memory-bound kernel, WHERE the data lives in the cache
// hierarchy dominates performance. Sweep the strip height / tile size and watch the
// L2 cliff.
//
// Hint: outer loop walks strips (oy0 += TH); inside a strip you run your full Stage-1/2
// loop, but restricted to rows [oy0, oy0+TH), doing every kernel tap before advancing.

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO: replace this placeholder with your tiled/blocked implementation.
    conv_naive(in, out, ker, H, W, K);
}
