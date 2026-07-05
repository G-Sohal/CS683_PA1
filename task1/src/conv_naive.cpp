// conv_naive.cpp — PROVIDED reference implementation.
//
// This is the correctness baseline AND the performance baseline that all speedups
// are measured against. Do NOT modify this file.
//
// The loop nest is the "textbook" one: for every output pixel, sweep the whole
// kernel and accumulate. The kernel loops are innermost, so the input is read with
// a strided, cache-unfriendly pattern and the compiler (with auto-vectorization
// pinned OFF, see the Makefile) emits plain scalar FMAs. Your job in the other
// stages is to do much better than this.

#include "convolution.h"

void conv_naive(const float* in, float* out, const float* ker,
                int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}
