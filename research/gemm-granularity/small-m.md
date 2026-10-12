# Small-M follow-up: can CK approach a weight-streaming ceiling?

User-directed next round: BF16 activations/weights, FP32 accumulation/output;
M=1,2,4,8,16,32,64. No further split-M. N remains both a representative
problem dimension and a split-N control. All work remains standalone.

## Recovery checkpoint

The network interruption left eight complete **nine-candidate** bring-up runs
(q=1..128), all with the same snapshotted executable SHA256
`5010dacba0a0558ce704cee5f5bc3a868c92bf4889500258f176b96bb92a6df4`.
Their logs contain no benchmark errors. The expanded build was incomplete:
its final object/benchmark link and padding tests were missing, and the
published executable was stale. Recovery inventoried and preserved that
state in `temp/gemm-granularity/small-m/recovery/`, rebuilt the whole closed
set, and ran the missing tests. The old bring-up is NOT pooled with the new
expanded-kernel measurements.

Screen jobs and confirmation jobs now write completion markers only after
successful process exit. Resume checks include the exact executable hash
and arguments. Partial CSVs from an interrupted job are rerun, not accepted
as completed results. The screen itself exceeded the tool timeout twice;
29 completed jobs were reused and the last incomplete job was rerun. All
30 planned jobs now have validated completion markers.

## What changed

1. **True one-launch split-K producer.** A single CK-backed grid uses z for
   K slices, with private FP32 partial outputs. It calls CK's device-level
   `UniversalGemmKernel::RunGemm` and supplies an explicit zero local split
   offset so CK does not apply blockIdx.z twice. One fixed-order SIMT combine
   follows. This is not q separate GEMM launches, and uses one HIP stream.
   The earlier separate-launch controls are retained and labeled separately.
2. Skinny one/two-wave tiles: 16x16x64, 16x32x64, 16x16x128,
   16x32x128 and 16x32x256, alongside the original seven configurations.
   There are **12 configurations**, each with regular and split producer
   entries; all use BF16 MFMA and FP32 C-shuffle/output.
3. Conventional affine weight storage with optional **128-BF16-element
   padding per weight row**: B's leading dimension becomes K+128, a 256-byte
   addition. This is BLAS-compatible leading-dimension padding, not custom
   permutation packing. Padding is initialized outside timing; logical
   input values are identical with and without padding. Leading dimensions
   and split offsets are independent: a K slice still uses the parent's
   padded leading dimension. Padding bytes are not counted as useful weights.

## Screening and confirmation

Native O3 HIP/C++20, HIP logical device 0 / PCI 0000:31:00.0 / gfx90a / 104 CUs,
CPU+memory NUMA node 0, no model/context, automatic clocks. Nominal HIP clock
fields are not live clock samples. No hardware traffic counters were collected.

- 42 parents: seven M values x N={2048,8192,14336} x K={4096,8192}.
- Split-K q={1,2,4,8,16,32,64,128}, one producer launch, one stream.
- Split-N q={1,2,4,8}, one or up to four literal HIP streams.
- B padding={0,128 BF16 elements}.
- Three screening rounds, eager/graph, isolated and batches of 16 complete
  groups. q=1 includes the rocBLAS control; fused q>1 is CK-only, explicitly
  logged rather than silently substituted with separate vendor launches.
- Each parent's selected warm graph-batch CK configuration was then fixed
  and rerun in **two fresh nine-round processes**. A third nine-round process
  checked single complete calls after the 64 MiB eviction-reader control.
  That pass is not a proven hardware cache flush. All these weights are
  16--224 MiB, larger than the GCD's nominal 8 MiB L2.

The contemporaneous 256 MiB checked reader median is **1405.420 useful GB/s**.
The rate below is `2*N_parent*K_parent / complete_group_time`: **useful
weight-byte bandwidth**, not actual HBM traffic. Activation/output traffic,
FP32 partial writes/reads and padding are excluded from the numerator;
producer, combine and coordination are included in the denominator.

### Confirmed small-M cases

Warm graph, 16 complete operations per interval, pooled from two nine-round
processes (18 samples per fixed configuration):

| M | Parent N,K | Selected CK tile | Split-K q | B pad | Weight GB/s |
|---|---|---|---:|---:|---:|
| 1 | 2048,8192 | 16x32x128 | 8 | 128 | 841.60 |
| 1 | 8192,8192 | 16x16x128 | 1 | 128 | 1183.99 |
| 1 | 14336,8192 | 16x16x128 | 8 | 128 | 1258.54 |
| 4 | 14336,8192 | 16x16x128 | 4 | 128 | 1235.27 |
| 8 | 14336,8192 | 16x32x128 | 1 | 128 | ~1226.5 |
| 16 | 14336,8192 | 16x32x128 | 1 | 128 | 1188.34 |
| 32 | 14336,8192 | 16x32x128 | N/2, one stream | 128 | 787.24 |
| 64 | 14336,8192 | 64x64x64 | 1 | 128 | 634.05 |

For M=1, the eviction-control single graph calls reach 708.50 / 1100.87 /
1235.70 GB/s at N=2048 / 8192 / 14336, respectively (K=8192). Do not exchange
these isolated latencies with batched rates. For the largest parent, a complete
operation reads 224 MiB of useful weights, and split-K uses K=1024 per slice.

**The high-bandwidth result is reproducible, but protocol matters:**

| M=1,N=14336,K=8192, q=8 | Process A | Process B |
|---|---:|---:|
| Eager, 16 complete operations | 1272.17 GB/s | 1274.38 GB/s |
| Graph, 16 complete operations | 1257.66 GB/s | 1259.48 GB/s |
| Eager, isolated call | 1256.85 GB/s | 1259.01 GB/s |
| Graph, isolated call | 1215.24 GB/s | 1208.24 GB/s |

An independent eviction-control run reaches 1269.90 eager / 1235.70 graph
GB/s for isolated calls. Eviction also exercises the memory system before
measurement; differences between these protocols are not uniquely attributed
to cache residency. Eager queued rates are **90.5--90.7%** of the measured
reader reference; graph queued rates are ~89.6%. This is evidence that CK
can operate near a streaming reference on large, small-M problems. It is
NOT an assertion that physical HBM traffic or every small GEMM hits 90%.

At the smaller 32 MiB parent, even the best confirmed configuration only
reaches ~60% of that reader reference. More splitting is not universally
better: q=64/128 generally adds too much short-tile/padding/partial work.
M=32/64 results fall well below the low-M plateau and need different kernel
feeding/reuse choices; weight-byte rate alone cannot diagnose their bottleneck.

### Matched layout and launch controls

Same M=1,N=14336,K=8192, same 16x16x128 tile, same q=8 and BF16 data:

- No B-row padding: graph-batch 738.55 / 742.19 GB/s in two nine-round runs
  (the initial broad screen was ~787 GB/s).
- +128 BF16 B-row padding: graph-batch 1257.66 / 1259.48 GB/s in confirmation.
  This is strong stride/layout sensitivity, not proof of channel mapping.
  Allocation/traversal/cache-state differences are not fully isolated.
- One fused producer: ~1258.94 GB/s in a separate nine-round control.
- Eight separate launches on one stream: ~889.13 GB/s.
- Eight separate launches across four streams: ~1161.63 GB/s.

The one-grid producer avoids conflating small-M split-K geometry with q
host launches or stream-fork overhead. All three controls include combine.
The unmodified rocBLAS q=1 control was ~512 GB/s without B padding and
~635--639 GB/s with padding in the three-round large-parent screen; that
control is not the production GEMV implementation.

## Correctness and ISA

Expanded all-output FP64 oracle tests pass for tail/padded K, N controls,
independent groups, q=3, padding=0/128, eager/captured execution, poisoned
outputs and canary guards. q=128 also passes repeated execution. Sampled
large-shape oracles and bitwise within-implementation checks pass throughout.
ISA inspection verifies native BF16 MFMA in all 12 regular and all 12 split
entries, four distinct register-reference instructions, and no scratch
instructions. No dependency-event scope was weakened.

Split-N output remains a strided parent-C view for M>1; communication packing
is not measured. No communication has been overlapped yet. For M=1, an N slice
is contiguous, but its actual collective/readiness timeline remains untested.

## Reproduce / durable artifacts

```sh
bash research/gemm-granularity/build.sh
python3 research/gemm-granularity/small_m.py temp/gemm-granularity/small-m/new-screen
python3 research/gemm-granularity/confirm_small_m.py \
  temp/gemm-granularity/small-m/new-screen/results.csv \
  temp/gemm-granularity/small-m/new-confirm --rounds 9
python3 research/gemm-granularity/report_small_m.py \
  --input temp/gemm-granularity/small-m/new-confirm/results.csv \
  --calibration temp/gemm-granularity/small-m/calibration/run.log \
  --out temp/gemm-granularity/small-m/new-summary
```

Options include `--k-launch separate|fused`, `--b-pad <BF16 elements>` and q
up to 128. A fused K producer requires axis=k and streams=1; it is not a
last-arriver/atomic-reduction implementation. Screen drivers never split M.

Raw data, exact executables, source snapshots and assembly are under
`temp/gemm-granularity/small-m/`: `recovery`, `bringup-q*`, `screen-a`,
`screen-summary`, `confirm-{a,b,evict}`, `confirmed-summary`, `calibration`,
`launch-*`, `pad0-*`, and `isa`. Curated tables are in `results/small-m-*`.
No production integration or system configuration changes were made.
