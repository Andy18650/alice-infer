# Study: BLAS scouting for lm_head, CPU + GPU (2026-07/09)

Question (user): can an off-the-shelf BLAS beat our hand-rolled kernels at
the lm_head GEMM shape — first on the Zen3 host, then on an MI250X GCD?
Two standalone benches, same shape throughout: W [V=128256, d=2048] bf16,
x [S, d] fp32, C [S, V] fp32 (the 1B's lm_head).

Both benches compare against the production oracle of their day —
`cpu_generic::matmul_wt` (bf16 weights, fp32 x, no conversion, half the
weight traffic of any fp32-BLAS path).

## CPU scout (`bench_lmhead.cpp`, commit 1d66638)

Reality check first: no obtainable Ubuntu OpenBLAS build (0.3.32, pthread,
LP64) ships `cblas_sbgemm` — fp32 `cblas_sgemm` was the only option, so
both BLAS paths pay a bf16→fp32 weight conversion, measured two ways:

- *convert*: conversion inside the timed call (the honest per-call cost)
- *cached*: weights pre-converted once (amortization ceiling; 2× memory)

| S    | OpenBLAS vs oracle                                   |
| ---- | ---------------------------------------------------- |
| 1    | **3× worse** — bandwidth-bound GEMV, 2× traffic      |
| 8    | 3.6× better only *cached*; loses with per-call conv  |
| 128  | 1.2 TFLOP/s cached — exactly ties our bf16-native `gemm_tile`, which needs no conversion and half the memory |

**Verdict: negative.** No BLAS integration on the CPU path; the bench
doubles as independent calibration that the hand-rolled `gemm_tile`
performs at mature-library level. (This is the origin of the "no-BLAS
verdict" in engine-design's retired decisions, scoped to CPU BLAS on
Zen3 — irrelevant to the device path, where vendor BLAS is used where
it fits.)

## GPU scout (`bench_lmhead_gpu.cpp`, commit ec58ada)

rocBLAS 5.1 `gemm_ex` (bf16 in, fp32 accumulate/HPA) on one GCD, host-only
C++ against the C APIs — no hipcc; Tensile ships pre-compiled gfx90a code
objects loaded at runtime. Per-phase timing: convert / H2D / gemm / D2H.

| S    | total (vs CPU oracle)   | gemm phase                |
| ---- | ----------------------- | ------------------------- |
| 1    | 1.00 ms (4.9× faster)   | 0.60 ms = 853 GB/s HBM    |
| 8    | 0.79 ms (77×)           |                           |
| 128  | 3.72 ms (168×)          | 68 TFLOP/s matrix-core    |

(rocBLAS 5.1 has no bf16 gemv; the n=1 GEMM is the S=1 path.)

**Verdict: strongly positive — this study is the seed of the
GPU-resident architecture.** Three findings that became engine doctrine:

1. **Pinned host buffers are mandatory.** Pageable H2D/D2H on this stack
   runs a driver-staged path (~1 GB/s measured here; later re-measured at
   2.2 GB/s vs 28.5 GB/s pinned in the loader context). Now §9 of the
   design: pinned staging for every weight load.
2. **The logits D2H is the offload tax.** A GPU lm_head wants the sampler
   on-device. Now §6: Sampler lives on the device; the sampled token id is
   the only regular per-step D2H.
3. **bf16 GEMM inputs cost max|Δ| 0.125 = bf16 epsilon** — covered by the
   tolerance-based equivalence policy (§10), the first datapoint of what
   became the bf16-GEMM-input budget on the canonical gates.

## Status (2026-09)

Concluded. The CPU path they scouted is retired (see engine-design
retired decisions); rocBLAS `gemm_ex` is the production GEMM path inside
`gpu_rocm` blocks. The benches are kept here as runnable evidence, outside
the engine build. Rebuild against the current tree with:

```sh
./build.sh [path/to/libopenblas.so]   # omit the arg to skip the CPU bench
```

(Both link the engine's `libalice.a` for the `cpu_generic::matmul_wt`
oracle; build the engine first, or point `ALICE_LIB` at another copy.)
