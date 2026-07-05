// matmul_reorder.cpp — STAGE 1: LOOP REORDERING / BLOCKING FOR REUSE   [TODO]
//
// The naive kernel computes one C[i][j] at a time, streaming a whole A row and a whole B
// row per output and reusing nothing. Reorganize the loops so each loaded A row and B row
// is REUSED across several outputs.
//
// Idea: compute a small block of C at once. For example, hold one A row (row i) and sweep
// it against several B rows (j, j+1, ...) before moving on — row i of A is then read once
// and reused across those outputs. Equivalently, interchange/split the i and j loops so the
// data you just loaded stays useful. This sets up the register- and cache-blocking of the
// later stages.
//
// Why it matters: matmul does O(M*N*K) work over O(M*K + N*K) data — there is enormous
// reuse to capture, and capturing it is the whole game. (Naive throws it all away.)
//
// Hint: keep the contiguous K dot product as the inner loop; change what surrounds it.

#include "matmul.h"

void matmul_reorder(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO: replace this placeholder with your reordered/blocked implementation.
    matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
