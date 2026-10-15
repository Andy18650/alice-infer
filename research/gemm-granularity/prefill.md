# Repeated prefill GEMM test of the shared tiled weight format

User-directed next round: test the fixed tiled layout on actual dense GEMM
operations, not only small-M/GEMV-like cases. BF16 weights/activations,
FP32 accumulation/output, beta=0. Standalone; no production changes.

## Scope and comparisons

42 shapes: M={64,128,256,512,1024,2048,4096}, with representative N,K:

- 2048,8192 (small output projection)
- 8192,8192 (generic wide projection)
- 14336,8192 (TP4-sized 70B gate/up)
- 8192,7168 (TP4-sized 70B down)
- 4096,14336 (8B down)
- 28672,4096 (8B gate/up)

M is the GEMM row count / nominal token count, not an attention context.
No model/context is loaded. q=1 throughout: **one GEMM per operation**, no
split-M/N/K and no communication. The storage contract is the same fixed
32x128 BF16 format tested by both CK and the diagnostic SIMT GEMV.

Layouts: plain affine, affine with 128-BF16/256-byte weight-row padding,
N-major storage tiles, K-major storage tiles. The closed dense shortlist is
CK 64x64x64, 64x128x64, 128x128x64, 128x256x64, 256x256x64. This is not an
exhaustive CK tuning search. rocBLAS is an affine-only vendor control.

Three-round screens select configurations. Two fresh nine-round confirmation
processes then test:

1. **Fixed geometry:** the plain-affine CK winner on all four layouts.
2. **Per-layout selection:** each layout's screened winning geometry, if
   different from the fixed geometry.
3. Both affine rocBLAS controls.

Confirmation jobs are grouped by configuration, have randomized job order
with different seeds between processes, and are resumable only for the exact
executable/arguments. They are matched-configuration repeats, not interleaved
per-round A/B measurements. The best confirmed affine CK reference includes
plain AND padded storage; comparing tiled only to an untuned/unpadded case
can exaggerate a practical benefit.

Native O3 HIP/C++20, one gfx90a GCD / HIP 0 / PCI 0000:31:00.0 / 104 CUs,
CPU+memory NUMA node 0, no model/attention context, automatic clocks. Nominal
clock fields are not live samples. Packing, allocation, initialization,
validation and graph capture are outside timing. All formats contain the
same synthetic BF16 values. Graph and eager, isolated calls and batches of
16 complete operations are separate. A third nine-round process tests
single calls after the 64 MiB eviction-reader control; it is not a proven
hardware cache flush. Every tested case passes sampled FP64 oracles, guard
checks and within-implementation repeatability.

## Main result: K-major tiled storage is mostly neutral for prefill

Warm graph, 16 complete GEMMs per timing interval, pooled from two independent
nine-round processes (18 samples/configuration):

| Comparison | Points | Min / median / max throughput change |
|---|---:|---:|
| tile-k vs plain CK, fixed geometry | 42 | -6.8% / +1.4% / +40.2% |
| tile-k, per-layout geometry vs best confirmed affine CK | 42 | -6.8% / +1.1% / +19.7% |
| same practical comparison, M>=128 | 36 | **-4.9% / +1.0% / +3.6%** |
| tile-n, per-layout geometry vs best confirmed affine CK | 42 | -73.3% / -29.4% / -6.7% |

These medians are across distinct shapes, not weighted model-level speedups.
Small ~1% differences should not be treated as decisive tuning wins. K-major
storage does **not** impose a general dense-GEMM penalty in this shortlist;
it is approximately neutral for the longer-M cases, with specific gains and
regressions. N-major tile order is consistently harmful here despite equally
contiguous storage tiles.

The practical M>=128 comparison also remains close to neutral for isolated
calls: warm graph median +1.3% (range -4.4%..+6.3%), eviction-control graph
median +0.9% (range -4.9%..+5.3%). These are separate protocols, not pooled
with queued throughput.

### Representative confirmed points

Same placement/context/protocol as above; useful GEMM TFLOP/s:

| M,N,K | Best confirmed affine CK | tile-k CK | Change | Best affine rocBLAS |
|---|---:|---:|---:|---:|
| 64,2048,8192 | 11.10 | 10.35 | -6.8% | 6.62 |
| 512,2048,8192 | 66.64 | 63.35 | -4.9% | see curated data |
| 512,14336,8192 | 77.80 | 80.42 | +3.4% | 103.22 |
| 1024,14336,8192 | 85.31 | 88.38 | +3.6% | 106.30 |
| 2048,14336,8192 | 87.01 | 88.31 | +1.5% | 93.17 |
| 2048,8192,7168 | 91.37 | 92.54 | +1.3% | 99.46 |
| 2048,4096,14336 | 100.17 | 100.73 | +0.6% | 97.00 |
| 2048,28672,4096 | 87.57 | 87.37 | -0.2% | 117.08 |
| 4096,8192,8192 | 94.06 | 96.47 | +2.6% | 117.68 |

The M=64 large gate/up point improves from ~40.69 to ~48.69 TFLOP/s versus
best confirmed affine CK (~+19.7%). This is a smaller-operation regime and
must not be generalized to large-M compute saturation.

### Why keep fixed-geometry and tuned comparisons separate?

At 1024x14336x8192, the same 128x128x64 CK geometry increases from plain
~76.14 to tiled ~88.38 TFLOP/s (+16.1%). But padded affine reaches ~85.31,
so the practical benefit over the best confirmed affine choice is only
~3.6%. Conversely, at 2048x8192x8192 the screened tiled winner uses a different
geometry, so some of its improvement is selection rather than just storage.
The curated tables preserve both comparisons explicitly.

The widest 256x256 packed candidate still has **76 private bytes/thread**;
its compiler resource/spill cost is included, not hidden. Winning smaller
geometries have zero private segments. A matched geometry changes its load
addressing/descriptor/codegen as well as the physical buffer, so it is not
an isolated proof of an intrinsic layout cost. No hardware-counter or channel
mapping attribution is made.

## Implication for the shared format

The K-major fixed tile format can feed both BF16 SIMT GEMV and BF16 MFMA GEMM.
It retains the previously confirmed small-M benefit (~1310 CK / ~1326 SIMT
useful weight GB/s on the large parent) without a broad prefill-GEMM collapse.
The trade-off is shape-specific, and the ~5% output-projection regression is
real enough to keep as a tuning target rather than dismiss it.

**This is a format viability result, not production adoption.** Dense CK
still trails rocBLAS on many shapes. That is a kernel/framework/pipeline
selection issue in this closed set, not automatically a storage-format
penalty: tiled and affine CK often have similar speed while both trail the
vendor. rocBLAS cannot directly consume the tiled format, so a production
choice would need faster tiled GEMM kernels, a separate affine representation,
or explicit repacking. Such costs and model-level effects are not measured.
No real-model/canonical integration, communication overlap or new minimum
compute-saturating shape is claimed.

## Reproduce / files

```sh
python3 research/gemm-granularity/prefill.py temp/gemm-granularity/prefill/new-screen \
  --suite screen --rounds 3
python3 research/gemm-granularity/prefill.py temp/gemm-granularity/prefill/new-confirm \
  --suite confirm --screen temp/gemm-granularity/prefill/new-screen/results.csv \
  --rounds 9 --seed 941
python3 research/gemm-granularity/report_prefill.py \
  --input temp/gemm-granularity/prefill/new-confirm/results.csv \
  --selection temp/gemm-granularity/prefill/new-confirm/selection.json \
  --out temp/gemm-granularity/prefill/new-summary
```

Current raw exact executables/source snapshots/logs/data are under
`temp/gemm-granularity/prefill/`: `screen`, `confirm-{a,b,evict}`, `summary`,
plus the completed background worker log and exit status. The executable is
byte-identical to the tiled-study executable snapshot at commit ffa02fb; only Python drivers and
analysis/documentation changed. No production sources or system configuration
were modified. Curated tables are in `results/prefill-*`.
