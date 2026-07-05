# CS683 PA-1 · Task 1 — Hardware-Conscious 2D Convolution

In this task you take a correct-but-slow 2D convolution and make it fast by
**thinking about the hardware**: the cache hierarchy, instruction-level parallelism,
and the SIMD (vector) units of a modern x86 core. You will apply four classic
techniques — **loop reordering, loop unrolling, cache tiling, and SIMD (AVX2)** — one
per stage, and measure what each one buys you.

The lesson is not just "make it fast" but *why* each transformation helps (or doesn't):
the same arithmetic, reorganized to match the machine, runs an order of magnitude faster.

---

## 1. The computation

A single-channel 2D convolution (cross-correlation — the kernel is **not** flipped) of a
`float32` image with a `K×K` kernel, in zero-padded **"same"** mode so the output is the
same size as the input:

```
out[y][x] = Σ_{ky=0..K-1} Σ_{kx=0..K-1}  in[y+ky][x+kx] · ker[ky][kx]
```

**Data layout (given to you; every stage uses the same signature):**

```cpp
// in : PADDED input, (H+2p) rows × (W+2p) cols, row-major, row stride = W+2p, p = K/2
//      The p-wide border is zero, so every access inside the loops is in-bounds —
//      you never need a boundary check.
// out: output, H × W, row-major, row stride = W
// ker: kernel, K × K, row-major
void conv_<stage>(const float* in, float* out, const float* ker,
                  int H, int W, int K);
```

`W` is always a multiple of 8, so an 8-wide (AVX2) inner loop over columns has **no
remainder tail** to special-case. `K` is odd (3 and 5 are tested).

---

## 2. What you implement

Edit **only** these five files in `src/`. Each already contains a detailed header
comment describing the technique and a hint. `conv_naive` is provided and is the
correctness reference — **do not modify it**.

| File | Technique | The idea in one line |
|------|-----------|----------------------|
| `src/conv_reorder.cpp`   | Loop reordering | Hoist the kernel loops out so the inner loop is a unit-stride stream over output columns. |
| `src/conv_unroll.cpp`    | Loop unrolling  | Unroll that inner loop to expose independent work (ILP) and hide FP latency. |
| `src/conv_tile.cpp`      | Cache tiling    | Block the output into cache-resident strips so the K·K passes hit cache, not DRAM. |
| `src/conv_simd.cpp`      | SIMD (AVX2)     | Do 8 columns per instruction with `_mm256_fmadd_ps` and a broadcast weight. |
| `src/conv_optimized.cpp` | **All combined** | Reorder + unroll + tile + AVX2, then tune. **This one is graded on speed.** |

The stages build on each other. A key teaching point lives in the first two stages:
the "obvious" reorder makes `K·K` passes over the whole (16 MB) output image, which does
**not** fit in cache — so at `K=3` it can run *as slow as or slower than* naive, and
unrolling a memory-bound loop barely helps. **Cache tiling (Stage 3) is what unlocks the
payoff.** Watch the speedup column tell that story.

---

## 3. Build and run

```bash
make        # builds ./bin/conv from your src/*.cpp
./bin/conv  # or: make run
```

`./bin/conv` (no arguments) runs a correctness smoke test, then prints a per-stage table
(correctness, time, GFLOP/s, speedup vs naive) for the graded `2048×2048, K=3` workload
and your autograder score, followed by an ungraded `K=5` table for reference.

### Checking one stage while you develop it

You do **not** have to finish everything before you can test. Each stub compiles from
the start (it just calls naive), so the project always builds, and you can check a single
stage in isolation — against the naive reference, on a workload you choose:

```bash
./bin/conv reorder              # naive vs reorder only, default 2048x2048 K=3
./bin/conv reorder 512 512 3    # custom H W K (W must be a multiple of 8, K odd)
./bin/conv reorder 512 512 3 42 # ... and a custom RNG seed (try several inputs)
./bin/conv all 1024 1024 5      # every stage on a custom workload (no score)
./bin/conv help                 # usage
```

The `correct` column compares your kernel's output to `conv_naive` on that exact input
(max abs error < `1e-3`), and the `speedup` column is measured against naive on the same
workload — so this is your fast edit-build-check loop. Changing the seed generates a
different random image/kernel, which is an easy way to gain confidence your kernel is
correct on more than one input. The final score always comes from the no-argument run.

### Compiler flags are pinned — do not change them

The `Makefile` compiles **every** stage with the identical flags:

```
-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma
```

- `-O2` — realistic scalar optimization (inlining, register allocation). Not a rigged `-O0`.
- `-fno-tree-vectorize` — the compiler's **auto-vectorizer is OFF**. This is essential:
  with it on (`-O3 -march=native`) the compiler auto-vectorizes the naive loop to
  **~55 GFLOP/s**, i.e. it does the SIMD work *for* you and there is nothing left to
  learn or measure. Off, the naive baseline is ~6 GFLOP/s and every speedup you see is
  the result of *your* transformation.
- `-mavx2 -mfma` — enable the AVX2 + FMA intrinsics your SIMD stages need (applied to all
  stages so none gets an accidental ISA advantage).

Do **not** add `-march=native`, `-O3`, or `-ftree-vectorize` — the autograder uses
exactly the flags above.

---

## 4. Grading (100 points)

Run `./bin/conv`; the score is printed at the bottom.

- **Correctness — 40 pts.** Each of the five stages must match the naive reference within
  `1e-3` (max absolute error). 8 pts per correct stage. An incorrect stage scores 0.
- **Speed — 60 pts.** The speedup of `conv_optimized` over `conv_naive` on the graded
  `2048×2048, K=3` workload, in tiers:

  | speedup | ≥ 2× | ≥ 4× | ≥ 6× | ≥ 8× |
  |---------|------|------|------|------|
  | points  | 15   | 30   | 45   | 60   |

  (`conv_optimized` must also be *correct* to earn any speed points.)

Timings use a warmup plus the median of several repetitions, so run on an otherwise-idle
machine for stable numbers. Absolute speedups vary by CPU; the reference solution reaches
~9× at `K=3` on a modern core.

---

## 5. Reference numbers (typical, one modern x86 core)

Your machine will differ, but the *shape* should look like this:

```
stage         speedup (K=3)   speedup (K=5)
naive             1.00×           1.00×
reorder           ~0.96×          ~0.90×     <- streaming shape, but poor locality
unroll            ~1.02×          ~0.96×     <- ILP can't fix a memory-bound loop
tile              ~1.6×           ~1.8×      <- tiling makes it cache-resident: the jump
simd              ~3.5×           ~3.9×      <- 8-wide AVX2 + FMA
optimized         ~9.5×           ~15×       <- + register blocking (accumulate in regs)
```

Notice tiling and SIMD help *more* at larger `K`: more kernel taps per byte of memory
traffic means more reuse to capture and more math to vectorize.

---

## 6. Submission

Submit your five edited files:
`src/conv_reorder.cpp`, `src/conv_unroll.cpp`, `src/conv_tile.cpp`,
`src/conv_simd.cpp`, `src/conv_optimized.cpp`. Do not modify any other file.

---

## 7. Going further (optional, not graded)

- Sweep the tile/strip size in `conv_tile` and plot the L1/L2 cliffs.
- In `conv_optimized`, try register blocking over more columns, `__restrict__`, aligned
  loads, or software prefetching — and measure whether each actually helps.
- Compile `conv_naive` alone with `-O3 -march=native` and see the auto-vectorizer nearly
  match your hand-tuned kernel. When does hand-written SIMD still beat the compiler?
- If `W` were *not* a multiple of 8, how would you handle the remainder columns?
