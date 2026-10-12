# Layout-first BF16-weight / FP32-activation GEMV

Standalone gfx90a implementation of the user-approved plan. **No production
engine or dispatch changes.** The first prototype reaches ~1400 GB/s on a
representative gate/up shape and ~1455 on a large, geometry-matched synthetic
matrix, including the combine where needed. This is not a universal GEMV
ceiling or a model decode-speed claim.

## Contract and format

Compute `y[n] = sum_k BF16(W[n,k]) * FP32(x[k])`, accumulating/outputting FP32.
There is **no activation truncation or BF16 conversion**. This intentionally
differs from the current engine's in-register BF16 operand policy. All weight,
x, partial and output buffers reside on the GPU; no allocations or transfers
occur inside measured operations. Native O3 HIP/C++20, no fast-math flag.

A block has 256 threads. Each thread owns two adjacent output rows and keeps
four independent accumulators for each. One 16-byte HIP uint4 record holds:

```text
word0: low BF16 = W[row0,k0], high BF16 = W[row1,k0]
word1: low BF16 = W[row0,k1], high BF16 = W[row1,k1]
word2: low BF16 = W[row0,k2], high BF16 = W[row1,k2]
word3: low BF16 = W[row0,k3], high BF16 = W[row1,k3]
```

Adjacent threads have adjacent records: 1 KiB per native wave load,
4 KiB per CTA tile. Each FP32 activation is reused for both rows. Dot products
need no cross-thread shuffle or LDS reduction. FP32 partials are stored in
`[split, padded_rows]` order; split=1 writes y directly without a combine.

Geometry:

```text
row_tiles = ceil(N / 512)
steps     = ceil(K / (4 * splits))
K_slice   = 4 * steps
job       = split * row_tiles + row_tile
pair      = row_tile * 256 + threadIdx.x
```

N is padded to 512 rows; K to `K_slice * splits`. Padded weights and activation
entries are zero. Tile/partition starts remain record-aligned, including
non-power-of-two split counts. Three physical layouts cover identical weights:

- **interleaved:** uint4 records `[step][job][thread]`;
- **adjacent:** uint4 records `[job][step][thread]`;
- **column:** uint32 row pairs `[k][row_pair]`, four coalesced scalar passes.

The two wide formats are split-geometry-specific: changing the split count
requires repacking. Column is an FP32-x control inspired by the existing
engine algorithm, **not the exact production numerical/kernel path**.
The CPU forward-scatter packer accepts a row-major BF16-bit span. The GPU
synthetic initializer independently uses inverse indexing; byte-for-byte
comparisons validate their agreement. Benchmark initialization generates
prepacked synthetic weights directly on the GPU to avoid timed packing or
large host copies. Packing cost is deliberately excluded under the study's
free-layout assumption. Real HF weights are not loaded by this harness.

## Kernel and pipeline

`partial<Format,Batch,Mode>` stages four/eight uint4 records per thread,
uses compiler scheduling boundaries between load/consume phases, then
expands BF16 and executes FP32 FMAs. It retains ordinary block/thread
indexing; no software wave partition is needed for reads or dot reductions.
Activations use uniform scalar-cache/SMEM loads. The compiler emits native
**v_pk_fma_f32** for row pairs: two FP32 FMAs, not matrix instructions or
BF16 activation math. The ordinary producer uses no MFMA, LDS, atomics or global synchronization.
The optional [last-arriver reduction](last-arriver.md) adds agent-scope atomic
counters, 4 B shared state and CTA barriers, eliminating the combine launch.
It is validated but not selected by default: net latency is mostly tied or
slower on the representative configurations.

A persistent/grid-stride CTA can handle multiple jobs. Natural-job grids
(one CTA/job) are also tested: merely capping at 80/104 can leave a small,
slow final wave of work. Candidate splits include unsplit, minimum fill,
double fill and a job count divisible by the target grid. The 80/104 starting
points are not a universal geometry law; natural controls may exceed 104.

The deterministic combine prefetches eight FP32 partials with independent
accumulators; up to seven leftover splits use a scalar tail. In particular,
three-split cases currently use that tail, so specialized small-split
combines remain a possible future improvement, not a completed optimization.

## First measurements

All numbers: native O3 HIP/C++20, gfx90a, HIP logical device 0 / PCI
0000:31:00.0, 104 CUs, CPU+memory NUMA node 0, **no model/context** or clock
overrides. Sampled active graphics/memory clocks 1700/1600 MHz. Matrices are
finite dense synthetic BF16 values with independently hashed rows/columns;
x is random FP32, generally not BF16 representable. Timed cases are at
least 32 MiB, beyond the GCD's L2; tiny correctness cases are not HBM claims.

Two independent quick sweeps, nine shuffled rounds, twelve operations/batch.
Fixed representative **complete GEMV** configurations, useful weight-byte GB/s:

| Shape N×K | Format | Batch | Splits | Grid/jobs | Run A / B |
|---|---|---:|---:|---:|---:|
| Synthetic 53248×8192 | adjacent | 4 | 1 | 104/104 | **1454.2 / 1455.9** |
| 70B TP4-shaped gup 14336×8192 | interleaved | 8 | 3 | 84/84 | **1394.5 / 1394.6** |
| 70B TP4-shaped o 2048×8192 | interleaved | 8 | 26 | 104/104 | **1020.9 / 1016.0** |
| Same gup, column 256-K control | column | 4 | 32 | 896/896 | 1278.0 / 1277.2 |

The synthetic shape deliberately has 104 row tiles and is 832 MiB: it tests
streaming saturation, not arbitrary shape optimality. Gate/up is 224 MiB and
o is 32 MiB. Useful traffic excludes padding, x and partial traffic; packed
weight-byte rate is separately reported. Neither is measured physical HBM
traffic. All three formats use the same nominal padded weight size for a
fixed geometry. Weight format plus geometry is compared above, not packing
alone with every other variable held constant.

Gate/up initially topped out ~1146 with the wide fixed-grid candidates;
using the natural 84-job grid recovers ~1394. This is consistent with work
balance being important; small-grid CU placement/tail time is not directly
traced. Eight staged wide loads remain before the first wait. The winning
GEMV uses 58 reported registers, capacity eight CTAs/CU, no scratch, no SGPR
lane or accumulator spill moves in its generated kernel.

## Single-operation protocol is reported separately

Two independent nine-round sweeps with one operation per event interval,
same fixed configurations (do not substitute a best-of-sweep winner):

| Shape/configuration | GPU GB/s, A / B | CPU wall us, A / B |
|---|---:|---:|
| Synthetic, adjacent/4/1/104 | 1420.3 / 1421.8 | 622.00 / 621.50 |
| TP4 gup, interleaved/8/3/84 | 1348.0 / 1361.8 | **181.85 / 179.86** |
| TP4 o, interleaved/8/26/104 | 835.5 / 825.7 | 47.58 / 48.03 |
| TP4 gup, column/4/32/896 | 1247.2 / 1257.9 | 195.47 / 194.69 |

GPU time for gemv_full includes the producer and combine launches on one
stream. CPU wall includes submission and final wait, excludes packing,
validation and warmup. Small-matrix launch/combine costs are material;
~1400 on a large operation must not be advertised for every shape.
Separately timed partial/combined phases are not necessarily additive
because event boundaries/cache/queue state differ. Judge the explicit
complete operation. Combine-only reports time and zero weight-bandwidth,
since it does not read weights.

## Decomposition and validation

Phases: read-only raw-word checksum, BF16 expansion/summation, GEMV partial,
complete GEMV and combine-only. Expand and partial timings exclude their
validation combine; complete GEMV includes it. Every timed batch is checked.

- CPU/GPU packed bytes agree for independent forward/inverse implementations.
- Odd N/K, N=1, empty K, split counts 1/3/7, partial tiles and padded K.
- Positive non-BF16-representable x with exact-one weights catches accidental
  activation quantization. Zero and cancellation inputs are included.
- FP64 per-row reference on the exact BF16 weight values and FP32 inputs.
  Every real output row must pass `abs_error <= 2e-4 + 3e-5*abs(reference)`.
- Bitwise repeated-output determinism in the final source; packing and
  dimension extents are checked, including arithmetic-overflow guards.
- Actual emitted batches: 4/8 wide loads or 16/32 scalar loads before the
  first wait in all modes. No scratch, cache-policy flags, register-lane
  spill or accumulator spill moves in these main kernels. Activation loads
  are SMEM; packed FP32 FMA is recorded explicitly by the inspector.

A three-round full-shape smoke adds 8B qkv/o/gup/down and TP4 qkv/down.
Best complete-operation screening medians: 8B down 1332, 8B gup 1302,
8B qkv 1164, 8B o 1039, TP4 down 1326, TP4 qkv 1163 GB/s. These are
provisional sweep winners, not nine-round repeated configuration claims.
Observed max FP64-reference absolute errors across the full smoke reach
~0.000298; every row also satisfies the stated relative/absolute gate.

## Reproduce / files

```sh
research/gemv-layout/build.sh
research/gemv-layout/run.sh temp/gemv-layout/new-test test 3 1
research/gemv-layout/run.sh temp/gemv-layout/new-quick quick 9 12
research/gemv-layout/run.sh temp/gemv-layout/new-latency quick 9 1
research/gemv-layout/run.sh temp/gemv-layout/new-model-smoke sweep 3 12
# Publication-policy controls (the ordinary producer/combiner is unchanged):
research/gemv-layout/run.sh temp/gemv-layout/new-last quick 9 1 release
research/gemv-layout/run.sh temp/gemv-layout/new-last-strong test 3 1 strong
```

`kernels.hpp`: private HIP kernels and packing definition.
`bench.hip`: RAM packer, FP64 oracle, tests and candidate sweep.
`check-codegen.py`: weight batches, waits, spill moves, FP32/packed-FP32 math.
`check-fused.py`: returned-value counter atomics, CTA barriers, publication
cache operations and preserved producer batches in the optional fused path.
`results/`: curated summaries/resources/ISA counts. Full source/binary,
raw rounds, generated assembly and telemetry: matching repo-local
`temp/gemv-layout/` folders. `bringup-a` precedes natural/balanced job grids;
its measurements are preserved separately, not relabeled.

Deferred: padded/dephased tile pitches, one-row variants, LDS activation
staging, fused/specialized combines, production-format/GEMM compatibility,
real-weight canonical/model integration. No new system dependencies and no
production source changes. Native builds/GPU tests are authoritative;
initial header LSP diagnostics used stale non-HIP context, while private
clangd configuration and CLI AST parsing resolve HIP correctly (remaining
CLI check failures are refactoring-tweak tests, not compiler diagnostics).
