// conv_unroll.cpp — STAGE 2: LOOP UNROLLING   [TODO: implement]
//
// Start from your reordered (AXPY) version, then unroll the inner column loop by a
// small factor (e.g. 8) so each iteration issues several independent load/FMA/store
// chains. With multiple independent chains in flight, the out-of-order engine can hide
// the multi-cycle latency of the floating-point adds (instruction-level parallelism).
//
// W is a multiple of 8, so an unroll factor that divides 8 needs no remainder handling.
//
// Heads up: unrolling attacks *compute/latency* limits, but your Stage-1 loop is still
// *bandwidth*-bound (it re-streams the whole output K*K times). So expect only a small
// gain here — a real lesson that unrolling alone cannot fix a memory-traffic problem.
// It pays off much more after Stage 3 makes the data cache-resident.
//
// Hint: keep the unit-stride inner loop from Stage 1; just process several columns per
// iteration.

#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO: replace this placeholder with your unrolled implementation.
    conv_naive(in, out, ker, H, W, K);
}
