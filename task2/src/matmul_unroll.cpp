// matmul_unroll.cpp — STAGE 2: LOOP UNROLLING   [TODO]
//
// Build on your Stage-1 blocking. Unroll to expose independent work so the out-of-order
// engine keeps several FMA chains and load ports busy at once:
//   - Unroll the K dot-product loop (accumulate into several partial sums to break the
//     single dependent add chain — FMA latency is multiple cycles).
//   - And/or compute a small register tile of C (e.g. 2x2 or 4x2 outputs) so the A and B
//     values you load are reused by several accumulators per iteration.
//
// Why it matters: a single dependent accumulator chain stalls on FMA latency; multiple
// independent accumulators hide it (instruction-level parallelism).
//
// Hint: several `float accXY` accumulators updated in the same K iteration is the core move.

#include "matmul.h"

void matmul_unroll(const float* A, const float* B, float* C,
                   int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO: replace this placeholder with your unrolled implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
