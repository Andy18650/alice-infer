# CTA grid 13x8, keeping four waves along N

User-directed equal-work mapping test. **Implemented only the 13x8 grid and
its natural M/N operand sharing.** The CTA's four waves remain 1x4 in MxN;
there is no 2x2 wave layout, LDS staging, cache-policy or backedge change.

**Result:** at D8/U8 the 13x8 grid reaches **91.060/92.179 FP8 TFLOP/s** versus
co-run 104x1 **81.989/81.850**, an 11.1%/12.6% improvement. Mean per-wave
intervals fall from ~115/~116 to ~103/~102 core cycles/MFMA. FP16 benefits
more strongly, reaching ~68.2/~68.6 versus ~41.5/~41.5 TFLOP/s.

## Mapping and what changes

`grid.x` splits N; `grid.y` splits M. Both launches have 104 CTAs, 256 threads
per CTA and 416 wave64s. Each wave owns a 32x32 output tile and traverses all
K. For block coordinates (x,y) and local wave w=0..3:

```text
A stream = y
B stream = 4*x + w
output M rows = 32*y .. 32*y+31
output N cols = 128*x+32*w .. 128*x+32*w+31
```

Thus the new grid has **eight distinct A streams**, each shared by 13 CTAs
(52 waves), and 52 B streams, each shared by eight M-group CTAs. Within a CTA,
all four waves share A addresses and have different B columns, exactly as
before. Every M group has distinct A data/addresses, not eight aliases of A0.
The shared-B interpretation is a proper GEMM: B does not depend on M.

| CTA grid (N,M) | Logical M | Logical N | K | A logical size | FP8 B logical size |
|---|---:|---:|---:|---:|---:|
| 104x1 | 32 | 13312 | 32768 | 2 MiB | 416 MiB |
| 13x8 | 256 | 1664 | 32768 | 16 MiB | 52 MiB |

Both compute **27,917,287,424 FLOPs/kernel** and 425,984 FP32 outputs. This is
an equal-work shape/mapping comparison, **not a speedup at unchanged M/N**.
At 8192 steps, K and input sizes double, with output sizes unchanged.

A/B retain their MFMA-packed layouts. A adds an outer M-tile dimension;
B is indexed by N-tile, independent of the M coordinate:

```text
A: [M-tile][K-step][lane][four FP16 values]
B: [N-tile][K-step][lane][four FP8 or FP16 values]
```

Both shapes use the **same allocations and addresses for common tiles**:
allocate maximum A for eight streams and maximum B for the old 416 streams.
The 104x1 case touches only A stream 0; 13x8 touches all eight A streams but
only the first 52 B streams. At 4096 steps, padded active A/FP8-B footprints
are 2.0049/417.0156 MiB for 104x1, 16.0391/52.1270 MiB for 13x8. The full
allocation remains large enough for either case; inactive pages are not read
by that case. No physical allocator-base change is mixed into the comparison.

The A data phase includes `3*M-tile`; B's phase uses N-tile, not flattened
output-wave ID. Independent CPU joint-phase counts validate both mappings.
This is synthetic changing data, not a model weight/activation capture.

## Same machine code, preserved load pipeline

**The two shapes launch the same compiled kernel entry** for each B format,
depth and unroll. Grid shape is selected by the host; there is no separate
shape-specific instruction body or new compiler scheduling policy.

The paired pipeline remains A-first/B-second, `vmcnt(2D-2)`, two ready A/B
banks and D=1/2/4/8 raw pairs. Wait and first read stay fused; queues continue
across backedges and receive a final endpoint drain. U=8/32 is diagnostic,
not a new dispatch or backedge optimization. No 2x2 wave arrangement is hidden
in the implementation.

Actual executable-object audit verifies all 16 native entries (32 launch
configurations): paired issue order/waits, three-lap pending-register safety,
24 rejected premature-read negative controls, final drains, zero private/LDS
bytes and no SGPR lane-spill transfers. At D8/U8, FP8 reports 56 VGPRs,
16 AGPRs, 31 SGPRs; FP16 68/16/28. Both grid shapes share these same entries.

## Protocol / numerical checks

Native HIP O3/C++20, gfx90a, system ROCm 7.1 / Clang 21.1.8. HIP device 0,
104-CU MI250X GCD, CPU+memory NUMA node 0, **no model/attention context**,
no clock/power changes. Both grids have 104 CTAs x 256 threads. Nominal
four waves/CU is not an enforced or measured CTA/CU placement trace.

Warm all entries and condition U=8 cases for >=2 seconds. Two independent
nine-round 4096-step processes (1729/2718), three rounds at 8192 (31415),
randomized case order, eight event-timed launches/sample. Last-launch
per-wave core/steady intervals provide cycles/MFMA and effective clock.
Core intervals include the final load drain but not an explicit MFMA-result
completion fence. GPU runs are sequential; CPU oracles/copies are outside
measurement. These are warm batched results, not cold-cache experiments.

A-only/B-only/both-change/restored fixtures are independently checked with
poisoned output/guards. All **384 fixture/case checks** and **672 timed samples**
match FP64 exactly. Bounded dyadic inputs retain the prior exact FP32 bound.
Short 32/96-step controls also pass. Earlier source/binary checkpoints remain
unchanged, and no engine/CMake code is modified.

## Results: D8/U8, K=32768

| B format | Grid | TFLOP/s, run 1 / run 2 | Core cycles/MFMA, run 1 / run 2 |
|---|---|---:|---:|
| FP8 | 104x1 | 81.989 / 81.850 | 114.909 / 115.713 |
| FP8 | **13x8** | **91.060 / 92.179** | **103.330 / 101.780** |
| FP16 | 104x1 | 41.518 / 41.456 | 245.366 / 244.289 |
| FP16 | **13x8** | **68.171 / 68.620** | **144.000 / 143.850** |

The corresponding gains are ~11–13% FP8, ~64–66% FP16. New-grid FP8 effective
clocks are ~1407/~1403 MHz versus ~1419/~1419 for 104x1; the improvement is
not a clock-boost artifact. FP16 similarly improves despite lower clocks.
No clock-response tuning is attempted.

### Bandwidth: cache-reused requests, not physical HBM

Median useful shader payload **TB/s**, D8/U8:

| B format | Grid | B requests, run 1 / run 2 | A requests, run 1 / run 2 | A+B requests, run 1 / run 2 |
|---|---|---:|---:|---:|
| FP8 | 104x1 | 1.281 / 1.279 | 2.562 / 2.558 | 3.843 / 3.837 |
| FP8 | 13x8 | **1.423 / 1.440** | **2.846 / 2.881** | **4.268 / 4.321** |
| FP16 | 104x1 | 1.297 / 1.296 | 1.297 / 1.296 | 2.595 / 2.591 |
| FP16 | 13x8 | **2.130 / 2.144** | **2.130 / 2.144** | **4.261 / 4.289** |

B now has eight-way reuse across M-group CTAs, so a rate above the GCD's
practical HBM ceiling is **not** a new HBM bandwidth record. These count all
wave requests, including those served by caches or merged misses. Both new
grid formats yield about 4.3 TB/s aggregate shader payload; this is a useful
observation, not a measured L1/L2/NoC ceiling.

For further accounting, CSVs include unique-matrix-byte equivalents:

```text
unique_B_equivalent = B_request_rate / grid_M
unique_A_equivalent = A_request_rate * grid_M / 416
```

For FP8 13x8 in run 1, these are **177.853 GB/s B + 54.724 GB/s A**, compared
with shader requests 1422.820 + 2845.640 GB/s. The equivalents are one logical
matrix pass divided by elapsed time, **also not actual HBM traffic**. Actual
external bytes depend on cache misses, eviction, merging and temporal reuse;
no hardware counters or NoC latency traces were collected. Priming/lookahead
request fields separately include `(steps+D+1)/steps` overhead.

### Other depths and larger K

FP8 U=8 old -> new TFLOP/s, run 1: D1 14.997 -> 17.063; D2 30.209 -> 33.209;
D4 56.387 -> 60.577; D8 81.989 -> 91.060. All runs/ranges are retained rather
than fitting a fixed backedge cost through memory variability.

At K=65536, D8/U8 FP8 is **93.544 TFLOP/s / 100.872 cycles/MFMA** for 13x8,
versus 81.771 / 117.065 for 104x1. FP16 is 67.700 / 149.111 versus
42.795 / 246.006. These larger controls have three rounds, not nine.
U=32 primary FP8 new-grid medians are ~90.688/~91.833 TFLOP/s; no new
unroll selection is made.

## Interpretation / stopping point

The requested geometry helps at equal work. But it changes **A fanout, B
reuse, active input footprints and logical M/N shape together**. It therefore
does not uniquely prove a NoC/controller-latency explanation. Physical CTA/CU
routing and cache hit rates remain unmeasured. Use the co-run 104x1 baseline,
not the preceding executable's absolute values, because the common indexing
code itself differs from that earlier probe.

**Round concluded.** Four waves stay entirely along N, no explicit cache/LDS
sharing is introduced, and the proposed 2x2 M/N wave arrangement is left for
the user's next discussion. Production code and earlier research kernels are
unchanged.

After this round, the user selected the [single-CTA latency diagnostic](latency.md)
rather than the 2x2 wave-layout route. The grid source/data above remain
unchanged; that follow-up compares cache protocols and depths through 16.

## Reproduce

```sh
bash research/fp8-w8a16/build-grid-probe.sh
python3 research/fp8-w8a16/audit_grid_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/grid/grid-probe 4096 9 1729
# Repeat: 4096 9 2718. Larger K: 8192 3 31415.
python3 research/fp8-w8a16/analyze_grid_probe.py
```

Raw CSV/logs, flags, hashes, audit, medians/comparisons are in `results/grid/`;
full ISA, loops, metadata and binaries are under `temp/fp8-w8a16/grid/`.
The analyzer verifies completeness, exact results and every request/unique-
equivalent accounting field. No engine regression rerun is claimed for this
isolated standalone experiment.
