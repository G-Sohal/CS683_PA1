# CS683 PA-1 · Task 2 — Hardware-Conscious Matrix Multiplication vs llama.cpp

In Task 1 you optimized a convolution. Here you optimize **matrix multiplication (SGEMM)**,
the workload at the heart of every neural network — and the one where **cache tiling and
software prefetching finally pay off** (a K×K convolution's reuse fit in cache; a large
matmul's does not). You will apply five techniques — **loop reordering / register blocking,
loop unrolling, SIMD (AVX2), cache tiling, and software prefetching** — then combine and tune
them. Finally you will **drop your kernel into a real llama.cpp build and run an actual LLM
inference**, comparing it against llama.cpp's own CPU matmul.

---

## 1. The computation

A dense single-precision matmul in the **"NT" layout** — the contraction dim `K` is the
**inner / contiguous** dimension of *both* operands. (This is exactly ggml/llama.cpp's
`ggml_mul_mat` layout, which is why your kernel can be injected into it unchanged.)

```
C[i][j] = Σ_{p=0..K-1}  A[i][p] · B[j][p]
```

- `A` is `M×K`, `B` is `N×K` (i.e. "B transposed"), `C` is `M×N`, all row-major.
- The inner K-loop is a **unit-stride dot product over both operands** — ideal for AVX2/FMA.

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

Edit **only** these six files in `src/`. `matmul_naive.cpp` is the provided reference — do not
modify it. Each stub has a detailed header comment and hints.

| File | Technique | Idea |
|------|-----------|------|
| `src/matmul_reorder.cpp`  | Loop reorder / register block | Compute a small tile of C so each loaded value is reused in registers. |
| `src/matmul_unroll.cpp`   | Loop unrolling  | Unroll the K loop for ILP (hide FMA latency). |
| `src/matmul_simd.cpp`     | SIMD (AVX2)     | Register-tiled 8-wide FMA dot products (un-blocked). |
| `src/matmul_tile.cpp`     | Cache tiling    | Block M/N so panels stay in L1/L2 — fixes the now memory-bound SIMD kernel. |
| `src/matmul_prefetch.cpp` | SW prefetch     | `_mm_prefetch` data ahead of use. |
| `src/matmul_optimized.cpp`| **All combined** | Tune it. **Graded**, and this is the kernel injected into llama.cpp. |

**Why SIMD comes before tiling.** This is the standard GEMM sequence. The scalar stages
(reorder/unroll) are *compute-bound*, so cache tiling does nothing for them. Only once SIMD
makes the arithmetic fast does the kernel become *memory-bound* — and *then* cache tiling is
decisive. The harness prints the stages in this order.

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
./bin/matmul simd                     # naive vs simd, default 1024³
./bin/matmul tile 1024 1024 1024       # custom M N K
./bin/matmul tile 1024 1024 1024 42    # ... and a custom RNG seed
./bin/matmul all 2048 2048 2048        # every stage at 2048 — watch tiling pull ahead of simd
./bin/matmul help
```

Each stage's result is checked against `matmul_naive` (relative tolerance), and its speedup is
measured against naive on the same workload.

### Pinned flags — do not change them

Same as Task 1: `-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma`. `-fno-tree-vectorize` keeps
the compiler from auto-vectorizing so the SIMD stage is *your* work; the flags are identical for
every stage so speedups reflect your technique, not compiler options. Do **not** add
`-march=native`/`-O3`/`-ftree-vectorize`.

---

## 4. Grading (100 points)

Run `./bin/matmul`; the score prints at the bottom.

- **Correctness — 30 pts.** Each of the six stages must match the reference (5 pts each).
- **Speed — 70 pts.** Speedup of `matmul_optimized` over `matmul_naive` on the graded 1024³
  workload, in tiers:

  | speedup | ≥ 5× | ≥ 10× | ≥ 18× | ≥ 25× |
  |---------|------|-------|-------|-------|
  | points  | 20   | 40    | 55    | 70    |

  (`matmul_optimized` must be correct to earn speed points.)

Absolute speedups vary by CPU; the reference solution reaches ~27× on a modern core. Run on an
idle machine for stable timings.

### Reference shape (one modern x86 core)

```
stage        speedup(1024³)   note
naive            1.00×
reorder          ~3.3×        register blocking
unroll           ~3.3×        + ILP
simd             ~21×         AVX2 register tile (memory-bound at large N)
tile             ~24×         + cache blocking  (at 2048³: ~17× -> ~21×, clearly bigger)
prefetch         ~22×         SW prefetch ~neutral here (strong HW prefetcher — a real lesson)
optimized        ~27×         tuned 4x3 micro-kernel + blocking
```

Note `prefetch` is roughly neutral on this hardware: modern cores prefetch regular streams in
hardware, so explicit prefetch mostly helps *latency-bound / irregular* access. Knowing when it
helps is the point.

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
3. downloads a tiny **F32** model (`stories260K.gguf`, ~1.2 MB — F32 so the F32 matmul path,
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

Submit the six `src/matmul_*.cpp` files you edited. Do not modify any other file. (`matmul_optimized`
is both graded and the kernel used by `make llama-demo`.)

---

## 7. Going further (optional)

- Bigger register micro-kernels / packing the A,B panels for the micro-kernel.
- A dedicated `M=1` (gemv) path so single-token generation is also vectorized.
- Multi-threading (OpenMP) — then compare against llama.cpp with matching `-t`.
- F16 / quantized weights (how llama.cpp really runs) — a much larger project.
