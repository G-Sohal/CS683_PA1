// conv_simd.cpp — STAGE 4: SIMD with AVX2 intrinsics   [TODO: implement]
//
// Vectorize the unit-stride inner loop over output columns using 256-bit AVX2
// registers, processing 8 float32 outputs per instruction.
//
// Per kernel tap (weight k = ker[ky][kx]), the AXPY inner loop becomes:
//
//     __m256 kv   = _mm256_set1_ps(k);                 // broadcast the weight
//     for (x = 0; x < W; x += 8) {
//         __m256 iv = _mm256_loadu_ps(&in_row[x]);     // 8 inputs
//         __m256 ov = _mm256_loadu_ps(&out_row[x]);    // 8 running sums
//         ov = _mm256_fmadd_ps(kv, iv, ov);            // ov += k * iv (fused)
//         _mm256_storeu_ps(&out_row[x], ov);
//     }
//
// W is a multiple of 8, so there is NO remainder/tail to handle. Combine this with
// your tiling from Stage 3 so the 8-wide streams also hit in cache.
//
// Why it matters: one core has vector units that do 8 (AVX2) FLOPs per lane per
// instruction. Scalar code leaves 7/8 of that throughput on the table. FMA also does
// the multiply and add in a single fused op.
//
// You need:  #include <immintrin.h>   (AVX2 + FMA are enabled by the Makefile flags.)
//
// Hint: keep out[] resident (Stage 3) and reuse the broadcasted weight across the row.

#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO: replace this placeholder with your AVX2 implementation.
    conv_naive(in, out, ker, H, W, K);
}
