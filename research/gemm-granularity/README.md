# BF16 GEMM granularity / saturation frontier (gfx90a)

Standalone research, user-authorized: **BF16 weights and activations, FP32
accumulation/output**. No production kernel, packing, dispatch, CMake, model,
or canonical-fixture changes. The installed CK/CK Tile and rocBLAS are used
through the normal native HIP/C++20 toolchain.

## Fixed tiled storage follow-up

See [tiled-weights.md](tiled-weights.md). The current prototype stores weights
in fixed 32x128 BF16 tiles, independent of split geometry. K-major outer order
feeds both a custom CK descriptor loader and a diagnostic BF16-input SIMT
GEMV from the same packed allocation. Confirmed large M=1 rates are ~1.31 TB/s
CK and ~1.33 TB/s SIMT, including combine; affine and scalarized-descriptor
controls remain preserved. An explicit, checked eight-BF16 alignment contract
was required to retain wide CK loads. This is not production promotion or a
change to the earlier full-FP32-activation GEMV contract.

## Small-M follow-up

The next round is recorded in [small-m.md](small-m.md): M=1..64, no further
split-M, one-grid split-K plus split-N controls, skinny one/two-wave CK tiles,
and affine B-row padding. Confirmed M=1,N=14336,K=8192 reaches ~1.27 TB/s of
useful weight bytes (~90.6% of the contemporaneous reader reference), including
FP32 combine. The expanded current set has 12 configurations; the seven-kernel
results below are the preserved first checkpoint, not silently relabeled.

## First checkpoint

A 1,200-shape grid and matched-parent partition probes are implemented and
measured. **A universal minimum (M,N,K) has NOT been established.** The first
important finding is that the tested dense kernels do not attain the
register-only BF16 MFMA reference; shrinking a kernel that is already below
that reference cannot establish the chip's absolute saturation threshold.

Release-equivalent native O3, single GCD HIP 0 / PCI 0000:31:00.0 (104 CUs),
`numactl --cpunodebind=0 --membind=0`, no model/attention context, automatic
clocks, ROCm/HIP 7.1.52801 / clang 21.1.8 / rocBLAS 5.1.0 / CK 7.1.1:

| Reference | Process A | Process B |
|---|---:|---:|
| Register-only BF16 MFMA, best grid median | 169.127 TFLOP/s | 169.201 TFLOP/s |
| 256 MiB reader, useful-byte median | 1411.560 GB/s | 1401.165 GB/s |
| Best measured large dense GEMM median | 118.375 TFLOP/s | 118.562 TFLOP/s |

Each calibration has nine samples; dense reference is graph execution with
32 complete operations per timing interval. These are **observed reference
envelopes**, not proofs of absolute hardware limits. Clock fields in logs are
HIP-reported nominal maxima, not sampled live clocks. Arithmetic reference
uses four distinct accumulator chains and no matrix input/output traffic
apart from a tiny checked final checksum; ISA verifies four native
`v_mfma_f32_16x16x16bf16_1k` instructions in its runtime loop.

The initial grid is M=1..2048, N=16..8192, K=16..8192 (powers of two).
Three rounds per configuration distinguish warm isolated calls, warm batches
of 16, and single calls after a 64 MiB eviction-reader pass. Eager and graph
results are separate. The eviction pass is outside both timers and is
**not a proven hardware cache flush**. A read-only bandwidth reference is not
an exact mixed read/write ceiling; no hardware traffic counters were collected.

For the large, compute-bound cases, 90% of the observed dense reference is
much weaker than 90% of the arithmetic reference. Initial warm graph-batch
winners near the dense knee, pooled from two independent nine-round process
runs (18 samples/configuration):

| M,N,K | Winner | Useful TFLOP/s |
|---|---|---:|
| 512,4096,4096 | rocBLAS | 83.70 |
| 1024,4096,4096 | rocBLAS | 85.79 |
| 1024,8192,4096 | rocBLAS | 106.49 |
| 2048,8192,4096 | rocBLAS | 108.62 |

1024x8192x4096 is near the 90%-of-dense boundary, not a robustly confirmed
90% pass. 2048x8192x4096 passes that weaker boundary but reaches only about
64% of the arithmetic reference. Low-K output-bandwidth cases must not be
advertised as MFMA compute saturation. Nor are warm-cache efficiencies HBM
measurements. The three-round grid is screening, not a variance-controlled
winner certification for every point.

### Matched work partitioning

Two process runs, five rounds each, graph, 16 complete groups per timing
interval, same GCD/NUMA placement, no model/context. Best closed-candidate
configuration per group; all K reductions included:

| Parent M,N,K | Partition / simultaneous streams | Group TFLOP/s |
|---|---|---:|
| 512,4096,4096 | unsplit N-control | 84.09 |
| same | N/2, 2 streams (piece 512x2048x4096) | 81.68 |
| same | N/4, 4 streams (piece 512x1024x4096) | 84.08 |
| 2048,8192,4096 | unsplit M-control | 108.70 |
| same | M/2, 2 streams (piece 1024x8192x4096) | 107.85 |
| same | M/4, 4 streams (piece 512x8192x4096) | 98.05 |
| 2048,8192,4096 | unsplit K-control | 108.72 |
| same | K/2, 2 streams (piece 2048x8192x2048), plus combine | 81.79 |

These are **aggregate group rates**, not individual chunk latency. N/4 on
the moderate parent preserves its observed unsplit throughput, even though
an individual piece need not saturate the GCD. Split-K loses substantially
once partial writes and the deterministic combine are counted. This is not
a universal verdict on K partitioning.

An additional five-round independent-matrix screen (graph, 16 groups per
interval) raises 32x1024x1024 from ~4.53 TFLOP/s alone to ~13.11 aggregate
TFLOP/s for q=16 on four streams; 16x4096x4096 rises from ~12.93 to ~18.32.
These are single-process screening results, not confirmed frontiers, and
neither establishes arithmetic saturation. Simply having more independent
pieces helps some cases but is not sufficient by itself.

**Communication has not yet been overlapped.** In particular, N slices write
strided views of the parent row-major C; RCCL would need compact chunk output
or packing. That cost is not in these measurements. Equal-FLOP partitioning
also changes reuse/traversal/algorithm selection, so differences are not
uniquely attributed to scheduling or occupancy.

## Contract, layouts, and traffic model

`C[M,N] = A[M,K] * B[K,N]`, beta=0. M=tokens, N=output features, K=reduction.
A is row-major BF16; B is column-major BF16 (equivalently row-major W[N,K]);
C is row-major FP32. Packing, input conversion, allocations, initialization,
validation and graph capture are outside timing. Thus this is not yet a
complete engine prefill path. No FP32 activation contract is implied.

Physical K is rounded to 16 and zero-filled in **both** implementations;
physical output row strides round N to 16. Useful FLOPs use real M,N,K.
CK additionally masks tile tails; `tile_flops` records rounded M/N/K tile
work, not measured instruction/counter FLOPs. Each candidate's actual
register/LDS/scratch/occupancy-query results and total CTA count are logged.
Unsupported CK shapes are explicitly logged/skipped (not silently sent to
rocBLAS). The fixed vector-output epilogue rejects some odd N values; the
vendor control still validates those shapes.

For one GEMM, optimistic compulsory useful bytes and useful FLOPs are:

```
F = 2 M N K
Q = 2 M K + 2 K N + 4 M N
I = F / Q
reference roof = min(P_register, B_read * I)
```

The script separately reports a roof using the observed dense plateau.
Read reference bytes are useful payload, **not physical HBM traffic**.
Both cache state and read/write mix limit how literally this memory model
can be interpreted. Efficiencies above one in cache-resident cases would
not establish an HBM-ceiling violation.

Partition layouts preserve matched parent data/work at aligned dimensions:

- M: separate A/C rows, shared B.
- N: shared A, separate B columns, strided slices of parent C.
- K: separate A/B reduction slices with parent leading dimensions; private
  FP32 partial outputs and one fixed-order SIMT sum into final C.
- independent: separate A/B/C matrices, supported for aggregate-fill probes.

K-partition compulsory bytes describe the mathematical parent. `modeled_bytes`
adds 8*q*M*N for partial writes/reads when q>1, in addition to final C. Storage
padding and partial capacity are reported separately. `padded_storage_bytes`
counts the tensors used by the operation, not the full allocation bill (it
excludes guards, vendor workspace, validation/eviction buffers and unused
reserve output storage).

## Kernels / validation / timing

Seven closed CK Tile basic-pipeline geometries, one BF16-only configuration
per TU: 16x64x64, 32x32x32, 64x64x64, 64x128x64, 128x128x64,
128x256x64, 256x256x64. Aligned eight-BF16 vector loads are explicit even
with K-tail masking. All use native BF16 MFMA and FP32 C-shuffle/output.
This is a small diagnostic set, not a claim of fully tuned CK pipelines.
rocBLAS `gemm_ex` is a separately labeled control, with atomics disallowed
and 64 MiB caller-owned workspace per stream; no vendor calls occur inside
CK device kernels. rocBLAS's private geometry is unknown and is logged as
zero rather than invented.

`test` verifies every real output against an independent FP64 oracle, tails,
zero K padding, overwritten-output guards, all partition axes, non-power-of-two
q=3, single/three streams, eager and captured execution. Timed shapes verify
up to 128 deterministic output coordinates (including first/last), guard
words and bitwise within-implementation repeatability. Cross-implementation
comparison remains tolerance-based (2e-4 + 3e-5*abs(reference)). Capture is
validated after poisoning output: capture itself cannot masquerade as an
executed result.

Multiple streams fork/join through standard HIP dependency events, including
split-K's consumer. An initial attempt to disable system fences on dependency
events produced NaNs in a multistream oracle test; standard event flags fixed
it. No reduced-scope publication proof is assumed. The benchmark's timing
events also retain standard flags. GPU times are complete event intervals,
not isolated instruction-body timings. CPU times include submission and final
wait. Repeated-operation averages must not be substituted for isolated-call
latency. Stream fork/join and combine overhead are included; graph construction
is excluded. No allocations or host transfers occur inside measured work.

## Reproduce

```sh
bash research/gemm-granularity/build.sh
bash research/gemm-granularity/run.sh temp/gemm-granularity/new-roof roof --rounds 9 --repeats 32
bash research/gemm-granularity/run.sh temp/gemm-granularity/new-map sweep --rounds 3 --repeats 16
bash research/gemm-granularity/run.sh temp/gemm-granularity/new-evict sweep --rounds 3 --repeats 1 --cache evict
python3 research/gemm-granularity/chunks.py temp/gemm-granularity/new-chunks --rounds 5
python3 research/gemm-granularity/summarize.py \
  --roof-csv temp/gemm-granularity/new-roof/results.csv \
  --roof-log temp/gemm-granularity/new-roof/run.log \
  --input temp/gemm-granularity/new-map/results.csv temp/gemm-granularity/new-evict/results.csv \
  --out temp/gemm-granularity/new-summary
```

Explicit `--m/--n/--k`, `--q`, `--axis m|n|k|independent`, `--streams`,
`--impl`, `--cache warm|evict`, `--mode eager|graph|both`,
`--k-launch separate|fused`, `--weights column|tile-n|tile-k`, and `--b-pad` are supported. New CSVs distinguish
producer-launch mode, padding and useful weight-byte rate; legacy summaries
remain readable as separate-launch, unpadded runs.
The default full grid has 1,200 points. Output is CSV; device/setup/checks
are in stderr. `run.sh` snapshots the executable/source and compiler/command
metadata under the chosen durable repo-local output directory.

First checkpoint raw data/assembly are under `temp/gemm-granularity/`:
`roof-final-{a,b}`, `map-a`, `map-evict`, `confirm-{a,b}`, `chunks-{a,b}`,
`summary`, `confirm-summary`, `chunks-pooled`, `independent-*`, and `isa`. Curated reference,
frontier, confirmation and matched-chunk data are in `results/`.

## Next steps

1. Improve the dense-kernel ceiling (optimized CK pipelines/layouts or better
   vendor choices) before asserting a physical minimum compute-saturating size.
2. Refine and repeat the M/N/K frontiers, including genuinely independent tiny
   groups and exact total-operation latency; do not infer per-chunk readiness
   from group throughput.
3. Compare compact communication-ready N outputs versus strided parent outputs.
4. Add real communication and measure readiness timelines/resource interference.
   Standalone saturation does not prove optimal computation/communication overlap.
