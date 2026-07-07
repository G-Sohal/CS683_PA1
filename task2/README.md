# CS683 PA-1 · Task 2  Hardware-Conscious Matrix Multiplication vs llama.cpp

In Task 1 you optimized a convolution. Here you optimize **matrix multiplication (SGEMM)**,
the workload at the heart of every neural network  and the one where **cache tiling and
software prefetching finally pay off** (a K×K convolution's reuse fit in cache; a large
matmul's does not). Task 2 has three stages  **SIMD (AVX2), cache blocking + software
prefetching, and combining every technique learned in this assignment (loop reordering /
register blocking, loop unrolling, SIMD, cache tiling, and prefetching)**  then combine and
tune it. Finally you will **drop your kernel into a real llama.cpp build and run an actual LLM
inference**, comparing it against llama.cpp's own CPU matmul.

---

## 1. The computation

A dense single-precision matmul in the **"NT" layout**  the contraction dim `K` is the
**inner / contiguous** dimension of *both* operands. (This is exactly ggml/llama.cpp's
`ggml_mul_mat` layout, which is why your kernel can be injected into it unchanged.)

```
C[i][j] = Σ_{p=0..K-1}  A[i][p] · B[j][p]
```

- `A` is `M×K`, `B` is `N×K` (i.e. "B transposed"), `C` is `M×N`, all row-major.
- The inner K-loop is a **unit-stride dot product over both operands**  ideal for AVX2/FMA.

Your single function is defined **with leading-dimension strides** so the *same code* works in
the microbenchmark (contiguous) and inside llama.cpp (tensor strides):

```cpp
// C[i*ldc + j] = Σ_p A[i*lda + p] * B[j*ldb + p]
void matmul_<stage>(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc);
```

FLOPs = `2·M·N·K`.

---

## 2. What you implement

Edit **only** these three files in `src/`. `matmul_naive.cpp` is the provided reference  do not
modify it. Each stub has a detailed header comment and hints.

| File | Technique | Idea |
|------|-----------|------|
| `src/matmul_simd.cpp`     | SIMD (AVX2) | Register-tiled 8-wide FMA dot products (un-blocked). Apply Task 1's loop reordering / register blocking and unrolling ideas here to build the micro-kernel. |
| `src/matmul_prefetch.cpp` | Cache tiling + SW prefetch | Block M/N so panels stay in L1/L2 (fixes the now memory-bound SIMD kernel), then add `_mm_prefetch` hints ahead of use. |
| `src/matmul_optimized.cpp`| **All combined** | Combine every technique from the assignment  reordering, unrolling, SIMD, tiling, prefetching  and tune it. **Graded**, and this is the kernel injected into llama.cpp. |

**Why SIMD comes before tiling.** This is the standard GEMM sequence. A scalar loop is
*compute-bound*, so cache tiling does nothing for it. Only once SIMD makes the arithmetic fast
does the kernel become *memory-bound*  and *then* cache tiling is decisive. That is why
`matmul_prefetch.cpp` asks you to add cache blocking to your Stage-1 SIMD kernel before
layering prefetch on top. The harness prints the stages in this order.

**Correctness for arbitrary shapes.** Your kernels (especially `optimized`) must handle any
`M/N/K` and any `lda/ldb/ldc`, because llama.cpp calls with its tensor strides and sizes.
Handle the remainder rows/cols/K with a scalar tail.

---

## 3. Build and run the microbenchmark

```bash
make            # builds ./bin/matmul
./bin/matmul    # correctness smoke test + graded 1024³ table + score
```

Check a single stage while developing (fast edit-build-check loop):

```bash
./bin/matmul simd                        # naive vs simd, default 1024³
./bin/matmul prefetch 1024 1024 1024     # custom M N K
./bin/matmul prefetch 1024 1024 1024 42  # ... and a custom RNG seed
./bin/matmul all 2048 2048 2048          # every stage at 2048  watch prefetch/optimized pull ahead of simd
./bin/matmul help
```

Each stage's result is checked against `matmul_naive` (relative tolerance), and its speedup is
measured against naive on the same workload.

### Pinned flags  do not change them

Same as Task 1: `-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma`. `-fno-tree-vectorize` keeps
the compiler from auto-vectorizing so the SIMD stage is *your* work; the flags are identical for
every stage so speedups reflect your technique, not compiler options. Do **not** add
`-march=native`/`-O3`/`-ftree-vectorize`.

---

## 5. Run your kernel inside llama.cpp

```bash
make llama-demo
```

This (see `llama/run_demo.sh`, single-threaded throughout):

1. clones a pinned llama.cpp (`b5731`) and builds `llama-cli` **stock**;
2. injects your `matmul_optimized` into ggml's CPU F32 matmul
   (`llama/inject.py` patches `ggml_compute_forward_mul_mat`; `llama/ggml_student_sgemm.cpp`
   adapts your kernel to ggml's tensor layout) and builds a **student** `llama-cli`;
3. downloads a tiny **F32** model (`stories260K.gguf`, ~1.2 MB  F32 so the F32 matmul path,
   the one you replaced, actually runs);
4. runs the same prompt on both builds (`-t 1`, fixed seed) and reports:
   - `[ok] student matmul_optimized was actually used` (the kernel really served ggml),
   - whether the **generated text matches** the stock build token-for-token (your kernel is
     correct inside a real model), and
   - **tokens/sec** for llama.cpp's native matmul vs yours.

Requirements: `git`, `cmake`, a C/C++ toolchain, `python3`, `curl`, and network access. Override
defaults with env vars, e.g. `LLAMA_TAG`, `MODEL_URL`, `PROMPT`, `NGEN`.

Note: single-token *generation* calls matmul with `M=1`, which your kernel handles via its scalar
tail (correct, but not its fast path); the *prompt-processing* matmuls are the batched, fast ones.
Compare the prompt-eval numbers to see your SIMD kernel shine.

---

## 6. Submission

Submit the three files you edited: `src/matmul_simd.cpp`, `src/matmul_prefetch.cpp`,
`src/matmul_optimized.cpp`. Do not modify any other file. (`matmul_optimized` is both graded
and the kernel used by `make llama-demo`.)

Additionally, submit a report (PDF) that includes the following plots and justifies the
observed performance gains or losses for each:

1. **Speedup vs. matrix size** — for each of `matmul_simd`, `matmul_prefetch`, and
   `matmul_optimized` over `matmul_naive`, varying `M=N=K` across a range of sizes (small
   enough to fit in cache up through sizes well beyond it, e.g. 128 up to 2048 or larger).
2. **Speedup vs. prefetch degree** — for `matmul_prefetch`, sweep the prefetch distance (how
   many floats/cache lines ahead you prefetch) and plot speedup over `matmul_naive` at each
   distance.
3. **Speedup vs. prefetch level** — for `matmul_prefetch`, compare the `_MM_HINT_T0`,
   `_MM_HINT_T1`, `_MM_HINT_T2`, and `_MM_HINT_NTA` locality hints and plot the resulting
   speedup for each.
4. **Speedup vs. SIMD width** — implement and compare 128-bit (SSE), 256-bit (AVX2), and
   512-bit (AVX-512) variants of your SIMD kernel, and plot the speedup each achieves over
   `matmul_naive`. Build these variants with the appropriate flags (e.g. `-msse4.2`, `-mavx2
   -mfma`, `-mavx512f`) for your own local measurements only  the graded submission of
   `src/matmul_simd.cpp` must still compile with the pinned flags above (AVX2), so keep any
   SSE/AVX-512 code in separate, ungraded files used only to produce this plot.