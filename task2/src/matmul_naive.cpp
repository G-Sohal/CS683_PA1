// matmul_naive.cpp — PROVIDED reference implementation.
//
// The correctness baseline AND the performance baseline that all speedups are measured
// against. Do NOT modify this file.
//
// Textbook triple loop: for each output element C[i][j], take the dot product of row i of
// A with row j of B (both K long, contiguous). It is correct and simple, but every C[i][j]
// re-reads a full A row and a full B row from memory with no reuse across neighbours, and
// (with auto-vectorization pinned OFF, see the Makefile) the dot product runs as scalar
// FMAs. Your job in the other stages is to do much better.

#include "matmul.h"

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
