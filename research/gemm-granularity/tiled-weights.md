# Fixed tiled weights for CK GEMM and SIMT GEMV

User-directed correction to the affine-layout baseline: physical HBM storage
is now tiled, not merely staged into tiles after strided global reads. Research
only; no production format/loader/kernel/dispatch or system changes.

## The storage contract

W is logically [N,K] BF16. Storage tiles are fixed **32 outputs x 128 K
positions = 8 KiB**. The tile dimensions and outer order are format metadata,
not launch parameters, and neither contains the split count.

Two lossless layouts hold the same values, with row-major data inside each
storage tile:

```
tile-n: [N_outer][K_outer][N_inside=32][K_inside=128]
tile-k: [K_outer][N_outer][N_inside=32][K_inside=128]
```

A 16x128 CK weight window accesses a contiguous 4 KiB half of an 8 KiB tile.
A 32x128 window reads the whole tile. Larger/smaller GEMM windows can span or
subdivide storage tiles; execution splits need not align to 128. N/K storage
tails are zero. For a full parent whose dimensions divide 32/128 there is
no storage overhead and no per-row 256-byte padding.

The CK loader presents the physical four-dimensional buffer as a logical
[N,K] tensor through merge/slice descriptors, then runs the same private
CK GEMM pipeline and FP32 epilogue. N/K submatrix origins are metadata;
packed pointers are not incorrectly treated as affine row pointers. Split-K
still uses one producer grid plus deterministic FP32 combine. Plain column
and +128-BF16-row-stride controls are retained. rocBLAS rejects tiled storage
explicitly in the harness: no silent repack or fallback is timed.

## A loader regression caught by ISA

The first correct descriptor implementation was slow: its dynamic slice
transforms lost vector-alignment information. Disassembly showed weight
`buffer_load_ushort` instructions where the affine kernel used wide loads.
For example, one packed entry had 64 ushort loads plus eight dwordx4 activation
loads. Numerical correctness alone did not expose this performance failure.

The current loader explicitly restores the **eight contiguous BF16 elements**
contract on the descriptor's top K dimension. This is valid because supported
K lengths/origins are multiples of eight, tile K=128 is divisible by eight,
and every storage-row/tile base has the required alignment. It does not assert
contiguity across an entire K-tile boundary. The support check rejects
misaligned origins. ISA verifies no ushort global/buffer weight loads in all
12 packed CK entries; loads are dwordx2/dwordx4 according to each distribution.
Native BF16 MFMA remains present. Kernel metadata confirms zero private
segments for the winning skinny kernels. The widest 256x256 packed entry
reports 76 private bytes/thread (512 registers); its scratch traffic is not
recognized by merely searching for `scratch_` mnemonics on gfx90a. The checker
now reads ELF private-segment metadata and requires an explicit 76-byte budget
for that packed candidate. This resource cost is retained in the dense controls,
not described as spill-free.

The scalarized implementation, data and executable are preserved separately
under `temp/gemm-granularity/tiled/`: `screen-a`, `dense-a`, `isa`, and
`scalarized-control`. They are not pooled with the corrected loader.

## Shared-format consumer / correctness

`simt.hip` is a diagnostic **BF16-activation/weight, FP32-accumulation/output**
GEMV: eight lanes per output row, four FP32 chains, wide uint4 reads and a
shuffle reduction. It consumes the *same packed allocation* as CK in M=1
cases without any repacking. This is not the earlier research GEMV's strict
FP32-activation implementation, and no claim is made about preserving that
implementation's optimum.

Independent packing tests compare a CPU forward-scatter packer with GPU
inverse initialization for odd dimensions and padded storage tails. The
same immutable packed blob at N=64,K=1008 is reused across CK q=1,2,3,7,9,21,63
schedules; several origins cross storage-tile boundaries. Blob bytes remain
unchanged. All-output FP64 oracles, q=3, N partitioning, independent groups,
canary guards, poisoned captured outputs and within-implementation repeats
pass for both tile orders. Thus changing split geometry does not require a
new weight pack. Synthetic per-slice padding in the legacy odd-K benchmark
is separate from the format contract; the immutable-parent test bypasses
that legacy convenience explicitly.

## Measurements

Native O3 HIP/C++20, gfx90a GCD 0 / PCI 0000:31:00.0 / 104 CUs, CPU+memory
NUMA node 0, no model/attention context, automatic clocks. Clock fields are
nominal HIP maxima, not live clock samples. Packing, allocation, initialization
and validation are outside timing. Complete-operation time includes producers,
FP32 combine and required coordination. Rates count **useful weight bytes**,
not physical HBM traffic or padding/activation/output/partial bytes.

Three-round screen: M={1,4,16,64}, N={2048,8192,14336}, K={4096,8192},
q={1,2,4,8,16,32}, four storage controls, twelve CK configurations, plus the
SIMT M=1 consumer and affine rocBLAS q=1 control. Graph isolated/batches of
16 are kept separate. The corrected screen is `aligned-screen`, not the
scalarized screen. Dense-GEMM controls are in `aligned-dense`.

Fixed configurations subsequently ran in two independent nine-round processes
(18 samples/configuration). The table uses warm graph batches of 16 complete
operations; it is not single-call latency:

| M,parent N,K | Consumer / fixed geometry | Padded affine | tile-k |
|---|---|---:|---:|
| 1,2048,8192 | CK 16x32x128, q=8 | 842.33 GB/s | 871.77 GB/s |
| 1,8192,8192 | CK 16x32x128, q=2 | 1180.97 | 1259.61 |
| 1,14336,8192 | CK 16x32x128, q=4 | 1228.39 | 1309.92 |
| 16,14336,8192 | CK 16x32x128, q=1 | 1192.59 | 1254.40 |
| 64,14336,8192 | CK 64x64x64, q=1 | 635.20 | 760.83 |
| 1,8192,8192 | SIMT GEMV, q=1 | 1217.39 | 1286.29 |
| 1,14336,8192 | SIMT GEMV, q=4 | 1258.71 | 1325.92 |

The previous best padded-affine CK geometry on the largest M=1 parent,
16x16x128 q=8, confirms at 1259.21 GB/s. Thus the best screened tile-k CK
geometry is ~4% faster than that tuned affine control; the matched-geometry
comparison is ~6.6%. Both are useful comparisons, not isolated format costs.
A fresh checked 256 MiB reader median is **1415.872 useful GB/s**: CK's
1309.92 is ~92.5%, and SIMT's 1325.92 is ~93.6% of that reference. These are
not assertions about actual physical memory-bus utilization.

Eviction-control isolated graph calls at the large M=1 parent confirm
1262.26 GB/s for tile-k CK versus 1202.30 for the same padded-affine geometry,
and 1294.54 for tile-k SIMT versus 1232.58 padded affine. A 64 MiB eviction
reader is not a proven cache flush; its work is completed outside both timers.
Warm queued results must not replace those isolated latencies.

### Contiguous tiles are not enough: outer order matters

Same M=1,N=14336,K=8192, CK 16x32x128 q=4:

| Storage | Confirmed weight GB/s |
|---|---:|
| Plain affine | 733.49 |
| +256 B affine row padding | 1228.39 |
| tile-n | 481.96 |
| tile-k | 1309.92 |

Both tiled formats have contiguous 8 KiB storage tiles. Their different outer
order changes the address stream of neighboring CTAs and successive K steps.
This is substantial traversal/layout sensitivity, not proof of a particular
physical bank/channel mapping. Some other tile-n geometries did better than
this matched geometry, but tile-k still won the screened small-M cases.

### Dense controls / limits

Three-round graph-batch screening (same placement/context, q=1), best closed
CK configuration per storage:

| M,N,K | Plain CK | Padded CK | tile-n CK | tile-k CK | Best affine rocBLAS |
|---|---:|---:|---:|---:|---:|
| 512,4096,4096 | 64.54 TF/s | 61.27 | 67.73 | 63.88 | 85.16 |
| 2048,8192,4096 | 94.60 | 92.36 | 68.08 | 92.69 | 108.68 |
| 4096,8192,8192 | 93.53 | 94.36 | 85.81 | 96.61 | 117.58 |

The shared fixed tile-k format is viable for both math families and does not
force a large regression in this CK set, but is not universally best. Dense
CK throughput remains below rocBLAS and far below the register-only MFMA
reference. This is not yet a fully optimized prefill implementation or a
proof of the minimum compute-saturating matrix size. Communication overlap,
compact communication output and real model/canonical integration remain
unmeasured. Storage-tile sizes and micro-layouts remain tunable choices.

## Reproduce / recovery

```sh
bash research/gemm-granularity/build.sh
numactl --cpunodebind=0 --membind=0 temp/gemm-granularity/bench --phase pack-test --rounds 1
python3 research/gemm-granularity/tiled.py temp/gemm-granularity/tiled/new-screen
python3 research/gemm-granularity/tiled.py temp/gemm-granularity/tiled/new-dense --suite dense
python3 research/gemm-granularity/confirm_tiled.py temp/gemm-granularity/tiled/new-confirm
# After extracting/disassembling the code objects, audit resource metadata too:
python3 research/gemm-granularity/check-codegen.py temp/gemm-granularity/tiled/aligned-isa \
  --candidates 12 --split-entries --packed-entries --allow-packed-private-bytes 76
```

`--weights column|tile-n|tile-k` selects storage. `--impl ck` restricts to the
MFMA candidates; `--impl simt_gemv` selects the M=1 SIMT consumer. Tiled formats
reject affine B-row padding. The format constants are in `weights.hpp`.

An interruption stopped the aligned rebuild after nine objects. Recovery
reused only objects newer than all local kernel/header inputs, with the exact
same compiler recipes, then built the missing objects and linked before
measuring. `resume-aligned.sh` and its background log/exit status are under
`temp/gemm-granularity/tiled/`; build/tests/screen/dense stages all completed.
No stale scalarized executable was used for the aligned measurements.

Raw binaries, sources, data and disassembly: `temp/gemm-granularity/tiled/`:
`aligned-screen`, `aligned-dense`, `aligned-isa`, `confirm-{a,b,evict}`,
`confirmed-summary`, `calibration`, `packing-test*`, `pack-aligned*`,
`aligned-test-*`, plus the preserved scalarized controls. Curated data are
under `results/tiled-*`.

Checkpoint validation: native standalone GEMV release/strong selftests, tiled
packing/reuse tests, all-output split-K oracles for all storage formats, ISA
checks (including the documented wide-packed private segment), shell/Python
syntax and `git diff --check` pass. The unchanged engine's
Debug ctest suite also passes (35 tests passed, optional CanonicalEnv skipped;
36 registered). Production sources/CMake were not modified or rebuilt for this
research checkpoint.
