# Standalone research log

This is the historical experiment/status record relocated from
[`docs/engine-design.md`](../docs/engine-design.md), followed by later research
updates. Paths in backticks are repository-relative; section references in
relocated entries refer to the engine design unless otherwise noted. Entries
retain their original observations, limitations and subsequent corrections,
including superseded results and historical artifact paths. For current replay
commands/layout, use the individual study README and active check scripts.

Research remains standalone. Detailed cooperative, geometry, traversal,
lookahead, register-tile, and vendor results plus limitations are recorded in
`research/p2p-allreduce/README.md` and `findings.md` (research commit ba8eb92).
Only the selected simple decode gather and vendor prefill are integrated;
other variants stay research candidates. Cross-implementation numerics are
tolerance-based; canonical gates and end-to-end measurements are in §17.
A standalone persistent counter probe now measures increment/wait progress
without payload or per-step launches (`research/p2p-allreduce/counter.md`).
At ws=8, parallel all-peer polling is ~1.81/2.07 us per cycle for relaxed /
release increments versus serial ~5.58/5.65; neighbor-only readiness is
~1.00/1.16. These are native O3, CPU+memory NUMA node 0, no attention context,
10,000 timed steps after 64 warm steps, nine samples. They are synchronization
skeleton costs, not one-way propagation or a claimed ring-AllReduce floor.
No production synchronization changes were made. A separate single-GCD
entry-timestamp study (`research/kernel-launch/`) records raw core and steady
clocks per wave with no entry barrier. Many CTAs begin close together below
residency, but zero-hold bodies can finish before later CTAs enter; 1664 CTAs
with a 100 us hold show ~102 us admission spread versus a kernel-specific
832-CTA capacity. This describes GPU scheduling, not a CPU per-CTA setup
loop or an exact hardware-allocation timestamp. At 256 one-CTA markers,
eager/graph start gaps are ~2.24/1.28 us (native O3, GCD 0, CPU+memory NUMA
node 0, no attention context). The extended identical-kernel cooperative
control is slower (~19.36 us start gaps versus normal graph ~1.60 for 256
one-CTA markers). Direct oversize cooperative launch rejects at the measured
832-CTA capacity, but captured replay accepts 1664 with device cooperative
context valid. The probe records this API inconsistency as unexpected, with
no grid.sync() and no valid timing claim; applications must enforce residency
before cooperative grid synchronization. No production launch changes made.

A standalone local VRAM ceiling sweep (`research/vram-ceiling/`) now exceeds
that earlier decode study's 1290 GB/s reference. Native O3 HIP/C++20,
HIP logical device 0 (PCI 0000:31:00.0, 104 CUs), CPU+memory NUMA node 0,
no model/context: 576 checked configurations vary load width, independent
chains, grid/block geometry and traversal. Two final process runs confirm
1 GiB best medians 1530.5/1509.3 GB/s; the same 8-byte-load configuration
sustains 1462.8/1460.2 GB/s medians at 4 GiB across nine randomized rounds.
Actual-code-object disassembly preserves wide loads; sampled active clocks
are 1700/1600 MHz core/memory without setting overrides. Treat ~1.46 TB/s
as a provisional multi-GiB local-read reference, not a GEMV/all-GCD ceiling;
size-dependent ranking and timing variance remain. An exact-code-object
occupancy query finds 59 VGPRs, no LDS/scratch, and 8×256-thread CTAs/CU
(832 across 104 CUs). A same-occupancy timestamped twin confirms all 3328
waves overlap at the 832-CTA launch, whereas a 1664-CTA control does not
fit the whole grid simultaneously. This is observed reproduction plus
capacity evidence, not a historical hardware-counter trace. No production
changes.

The follow-up `research/vram-addressing/` probes fixed-payload tile strides,
base alignment and disjoint paired streams on that same GCD/placement,
without a model/context. Native O3, 256 B tiles, 32 MiB useful reads:
8 KiB stride is ~1100 GB/s; 16/32/64 KiB fall to ~889/557/304, while +256 B
padding restores ~1105/1085/1070. Seven randomized rounds and independent
32/64 MiB process repeats retain the losses. Paired streams show periodic
half-stride relief and a reproducible difference between 8/16/24 GiB plane
gaps. These are effective contention signatures, not proof of a 256 B
physical HBM interleave or an XOR swizzle. Counter collection is blocked by
AQL-profile extension initialization; no system workaround or production
change was made. See the study for codegen, checksum and mapping caveats.

`research/vram-ilp/` now separates addressing from compiler scheduling at
104×256 threads on that same GCD/NUMA placement, native O3, no model/context.
The preserved U=16 global reader emits load→vmcnt(0)→consume repeatedly;
U=8 batches loads. A scheduling-constrained U=16 reader keeps addresses
unchanged and restores ~559→1396 GB/s at 1 GiB (nine rounds of 12 reads).
Wave-local 4 KiB tiles reach repeated ~1480 GB/s medians at 1 GiB and
~1420 at 4 GiB; one 1502.9 median repeats at 1486.4, not a stable 1500+
claim. A timestamped twin places one CTA on each of 104 CUs with four
waves/CU. Larger unrolls, launch bounds and ping-pong banks are not uniform
wins. Matched unchanged 832-CTA controls run ~1400 here, below their earlier
1530.5; that cross-campaign shift is not isolated. No production changes.

`research/vram-coalescing/` compares equal 256-dword wave tiles using A:
per-lane 16 B vector loads, B: four adjacent-lane scalar passes, and C:
four per-lane contiguous scalar loads. Same GCD/NUMA placement, native O3,
no model/context, 1 GiB, 104×256, four independent tiles: A/B repeat at
~1430 GB/s, controlled scalar C ~676. B/C have identical scalar load counts,
graduated waits, resource capacity and cache flags; more batching or 832
CTAs does not rescue C. Normal LLVM optimization widens C into A's 16 B
loads and restores ~1430. This isolates instruction/address-pattern
sensitivity, not physical HBM traffic or a channel map. A same-capacity
hardware-ID/timestamp follow-up (`research/vram-coalescing/packing.md`)
finds identical CU/SIMD spreading for A/B/C: 104/208/416/832 CTAs give
1/2/4/8 overlapping CTAs per CU, each CTA's four waves split one per SIMD.
A 1664-CTA control peaks at 3328 waves, with later assignment as work
finishes. Two process runs × seven samples confirm this workload's balanced
packing, not a general memory-contention-minimization policy. The current
probe now uses raw uint32_t input and HIP uint4, with no float bitcasts or
custom vector aliases. Revalidation retains the A/B ~1430 versus controlled
C ~675 GB/s pattern and the same CU/SIMD packing; historical float-bit
snapshots remain separately labeled. A uint4 expression can split into two
64-bit loads in the controlled build, which the ISA checker records rather
than treating the source type as an instruction guarantee. Barrier removal
is now the default research path, with separate compiler-barrier controls:
A/chunks=8 and optimized C/chunks=4 return to serial load/wait/use and slow
to ~1020 GB/s, while A/B at four chunks retain batching. The narrowed
barrier-free A/B four-chunk test repeats at grids 104/208/416: 1 GiB A is
~1435/1355/1309 GB/s, B ~1444/1417/1371; more blocks do not improve the
winner. These are native O3, same GCD/NUMA placement, no model/context,
two process runs of nine rounds; grid count also changes address spacing.
See `research/vram-coalescing/tuning.md`. A rotating register-buffer
follow-up (`ring.md`) finds that unconstrained C++ is rebatch-scheduled;
per-slot compiler constraints retain steady waits at vmcnt(3) for A and
15..12 for B, with no steady-state drain to zero. Ordered B depth 4 at
104×256 repeats at 1470.3/1465.6 GB/s versus batch B 1426.6/1434.9 at
1 GiB; at 4 GiB 1423.7/1424.0 versus 1401.8/1398.9, winning all paired
rounds. A does not improve. The exact historical uint2/U16 code object,
interleaved on the same input, still runs ~1400 rather than 1530. These
are native O3, same GPU/NUMA placement, no model/context. The small B gain
retains the experiment rather than triggering the requested failure/revert
path; the original source remains unchanged for inspection. A block/thread
indexing follow-up (`indexing.md`) keeps the A/B four-chunk, 104×256 rates
within ~1% of wave controls at 1/4 GiB (~1425/~1436 and ~1399/~1386 GB/s),
so the research baseline now uses block/thread IDs for loads. B uses four
block-wide stripes; A/C preserve their addresses. Wave arithmetic remains
only in the checksum and a separate historical control; the ring experiment
retains its original wave mapping. The smaller-grid sweep (`small-grid.md`),
with unchanged GPU code and A/B four-chunk baselines, finds 80 CTAs retain
near-full throughput (1 GiB A 1461/1455, B 1442/1442 GB/s); 64 decline to
~1344/~1317 and 32 to ~1002/~827. Two processes, nine shuffled rounds,
1/2/4 GiB, same GPU/NUMA placement and no model/context. Grid-stride changes
alongside concurrency, so this is not proof of an intrinsic active-CU knee.
Increasing the staged window with compiler constraints (`inflight.md`)
recovers A/64/chunks8 near-full (~1.4 TB/s), and A/32/chunks16 to
~1.35 TB/s at 1 GiB / ~1.3 at 4 GiB. B peaks at chunks8 and regresses at
16. ISA verifies full 4/8/16 wide or 16/32/64 scalar load batches before
consumption waits; B16 has heavy SGPR-to-VGPR lane spilling despite zero
scratch. Phase-only constraints shorten code but do not eliminate the
spills. Flipping the chunk/grid axes (`chunk-order.md`) stages adjacent
CTA tiles instead of grid-spaced tiles: phase-only controls repeat A/64/
chunks4 at 1420/1423 versus 1339/1331 GB/s, but A/64/chunks8 falls to
1359/1361 from 1417/1425. B/32/chunks16 improves 713/714 -> 1049/1063;
its old hot-loop SGPR lane-spill code disappears, with higher reported
register allocation instead. Both layouts retain complete load batches,
so this is an address-order plus codegen/resource comparison, not isolated
cache or channel evidence. Four nine-round 1/2/4 GiB runs cover phase-only
and per-load constraints; expanded partial-record/group tail checks pass.
The full 36-case A/B × 104/80/64 × 4/8/16 × both-order matrix
(`order-matrix.md`) is now measured in two new phase-constrained process
runs (nine rounds each, 1/2/4 GiB), with GPU code identical to the prior
order smoke. Pooled 1 GiB medians favor adjacent/four chunks at 104/80
for A (~1469/1461 GB/s); A/64 favors grid/eight (~1438). B/104's larger
batches catch up at 4 GiB (~1387/1397 adjacent chunks8/16). All partial-
record/group and timed checks pass; no grids above 104 are in this matrix.
A same-allocation 1 GiB-window + sustained-warmup study (`windows.md`)
finds no consistently slow quarter: adjacent A/104/chunks4 individual
windows remain ~1440-1480 versus full 4 GiB ~1409; four short kernels
recover part (~1430), not all, of the difference. B/104/chunks16 grid keeps
the opposite size effect. Two rich-clock repeats, a basic-monitor control
and a 30 s bandwidth timeline show no progressive decline. Firmware current
GFX/SoC/memory reports 1700/1090/1600 MHz; averages ~1600/999/1599 remain
similar across sizes/stages, with zero reported throttle masks. FCLK remains
0 in a deep-sleep DPM entry even under load, so hidden fabric changes are
not excluded. Window restarts also change traversal/CTA assignment and
translation/cache reuse; no unique hardware cause is identified.
A four-body-in-one-grid control (`body-control.md`) now measures equal
4 GiB payloads including launches, reductions and completion boundaries.
Runtime outer loops are not unrolled and add no grid/CTA synchronization.
A's four-kernel streaming path saves ~1-2% net wall time, while single-
operation paired latency wins are irregular; repeating bodies in one grid
mostly does not recover A's short-window rate. B is layout/chunk dependent.
Inlining changes resources (B/adjacent/chunks16 reports 262 -> 400 regs
with more accumulator moves, still zero scratch), so this is not pure
kernel-boundary causality. Six processes cover twelve-operation throughput
and one-operation latency, with all separate-output/tail checks passing.
The user closes this kernel-loop campaign here and sets VRAM layout as the
next standalone research focus: multiple well-coalesced, sufficiently
batched readers reach ~1350–1450 GB/s, and the remaining small differences
do not by themselves justify production fission/dispatch changes. This is
a research priority, not a unique hardware-bottleneck attribution. A durable
post-power-cycle ceiling replay note now records the byte-identical old
executable's ~1406–1413 GB/s versus its historical 1530 confirmation.
The barrier-free baseline and production selection remain unchanged.

The next standalone study, `research/gemv-layout/`, implements a layout-first
SIMT GEMV: BF16 weights, genuinely FP32 x/accumulation/y, two output rows
per thread, uint4 records across four K positions, and four/eight staged
records. Interleaved/adjacent packing and an FP32-x column-major control
share deterministic split-K plus combine; packing/upload are outside timing.
Native O3, same GCD/NUMA placement, no model/context: two nine-round
12-operation sweeps repeat complete GEMV at 1454.2/1455.9 GB/s for a large
104-row-tile synthetic shape and 1394.5/1394.6 for the 70B TP4-shaped gup
14336×8192 (interleaved, batch8, splits3, natural 84-job grid). Merely capping
persistent grids can leave a slow last wave of jobs. The smaller o shape
2048×8192 reaches 1020.9/1016.0 end-to-end; isolated one-operation sweeps
reduce gup to 1348/1362 GPU GB/s with ~180–182 us CPU wall time. All rows
pass FP64-reference gates, byte-exact independent packing checks and final
bitwise determinism tests, including odd N/K, empty K, cancellation and
FP32-x precision. ISA retains full weight batches and uses native packed
FP32 FMA plus SMEM x loads, with no scratch/register spilling. A three-round
full-model-shape smoke is provisional; no real model/canonical integration
or production packing/dispatch change is made. A tile-local last-arriver
follow-up (`last-arriver.md`) publishes each split, decrements an agent-scope
counter and lets the old-count-one CTA reduce fixed-order FP32 partials.
Non-last persistent CTAs continue rather than wait/return; counters self-reset
for stream-ordered reuse. Strong and release-publication protocols pass
bitwise reference, changed-input/repeated-call, grid=1 and oversubscribed
2050-CTA tests. Release-only producer ordering avoids non-last acquire cache
invalidation; 256 B counter spacing is a separate control. Two nine-round
throughput/isolated-call repeats find no consistent net gup win; the small
o projection is ~1-2 us slower in isolated CPU wall time. Fences/atomic/CTA
reduction costs outweigh the saved launch here. The two-kernel path remains
the control/default; no production synchronization change is made.

**BF16 prefill granularity research (2026-10, user-directed):** standalone
`research/gemm-granularity/` measures BF16 activations/weights with FP32
accumulation/output, a seven-geometry CK Tile set and rocBLAS control. Native
O3, one gfx90a GCD (HIP 0), CPU+memory NUMA node 0, no model/context: two
nine-sample calibration runs reach ~169.1/169.2 TFLOP/s in a checked
register-only MFMA loop versus ~118.4/118.6 in large dense GEMMs; a checked
256 MiB reader reaches ~1412/1401 useful GB/s. The latter is read-only payload,
not physical traffic or a mixed read/write ceiling. A 1,200-point shape grid
separates warm/eviction-control, isolated/batched and eager/graph protocols;
selected knees repeat in two nine-round processes. No universal minimum
compute-saturating (M,N,K) is established: dense kernels still fall well below
the arithmetic reference. Matched-parent two-process, five-round partition
probes include fork/join and split-K combine. At parent 512x4096x4096, N/4 on
four streams retains ~84 TFLOP/s group throughput; at 2048x8192x4096, M/2
retains ~108 of ~109, while K/2 plus combine falls to ~82. These are aggregate
rates, not chunk readiness or communication-overlap results; N output slices
are currently strided and communication packing is not included. All-output
small-case FP64 oracles, tails/guards, q=3, one/three streams, poisoned captured
outputs and within-implementation repeat checks pass. ISA confirms native
BF16 MFMA and no scratch in the closed CK set. No production integration or
execution/packing policy change is made; the FP32 stream contract is unchanged.

**Small-M CK follow-up (2026-10, user-directed):** M=1..64, representative
N=2048/8192/14336 and K=4096/8192, no further split-M. The standalone set now
has 12 BF16 configurations, including one/two-wave skinny tiles and wider K
windows. A true one-launch CK split-K producer covers slices in grid.z, writes
private FP32 partials and uses a deterministic SIMT combine; separate launches
and split-N streams remain labeled controls. Affine weight-row padding of 128
BF16 elements preserves the same logical weights and remains BLAS-compatible.
Native O3, one GCD HIP 0, CPU+memory NUMA node 0, no model/context: two nine-round
confirmations at M=1,N=14336,K=8192, tile 16x16x128, q=8 and padded weight stride
reach 1272/1274 useful weight GB/s eager queued and 1258/1259 graph queued,
including combine. The contemporaneous 256 MiB reader is ~1405 useful GB/s;
these are ~90.6% / ~89.6% payload-reference rates, not physical HBM traffic.
An eviction-control isolated run reaches ~1270 eager / ~1236 graph GB/s.
The smaller 32 MiB parent only reaches ~842 queued / ~709 isolated graph GB/s.
Same-kernel unpadded controls are ~739/742 GB/s, demonstrating layout/stride
sensitivity without proving channel mapping. The network-interrupted expanded
build was recovered without losing earlier executable snapshots; incomplete
jobs are rerun using hash-bound completion markers. All-output small-case
FP64 oracles, padding/tails, q=3/128, guards, captured output poisoning and
within-implementation repeat checks pass. ISA verifies BF16 MFMA in all 24
regular/split entries with no scratch instructions. No communication overlap,
production changes, system configuration changes or weaker fence policy is
introduced. Detailed protocols and limitations are in the research small-M note.

**Fixed tiled weight storage (2026-10, user-directed):** the standalone
harness now has a split-independent 32x128 BF16 storage tile, with N-major
and K-major outer orders. A custom CK tensor descriptor exposes tiled weights
without a timed global repack; a BF16-input SIMT GEMV consumes the same packed
allocation. The first correct sliced descriptor scalarized weight reads;
assembly showed ushort loads. An explicit eight-BF16 K-vector alignment
contract, checked against supported origins/lengths, restores wide loads.
ELF metadata reports zero private bytes for the skinny winners; the widest
256x256 packed variant reports 76 private bytes/thread, missed by a mnemonic-
only scratch check and now explicitly recorded by the resource-aware inspector.
Independent CPU-forward/GPU-inverse packing tests, tails/guards and reuse of
one immutable blob across q=1/2/3/7/9/21/63 pass. Native O3, one gfx90a GCD,
CPU+memory NUMA node 0, no model/context: two nine-round confirmations at
M=1,N=14336,K=8192 give ~1310 useful weight GB/s for tile-k CK 16x32x128 q=4
versus ~1228 for matched padded-affine geometry (~1259 for the previous tuned
affine winner), and ~1326 for SIMT versus ~1259 padded affine. These complete
rates include FP32 combine and are ~92.5% / ~93.6% of a contemporaneous
~1416 useful-GB/s reader reference, not physical HBM traffic. Tile-n is much
slower despite contiguous storage tiles, showing outer-order sensitivity
without establishing a channel map. Three-round dense CK controls show
mixed small gains/losses versus affine CK and remain below rocBLAS; no minimum
compute-saturating shape, actual communication overlap or real-model speedup
is claimed. No production changes or narrowing of the earlier full-FP32-x
GEMV study are made. Details and retained scalarized controls are in the tiled
weight research note.

**Repeated prefill GEMM storage test (2026-10, user-directed):** 42 standalone
BF16-input/FP32-output shapes span M=64..4096 and representative projection /
8B and TP4-sized 70B FFN N/K dimensions. Five dense CK geometries and affine
rocBLAS controls are screened, then the plain CK winner is held fixed across
layouts while per-layout winners are confirmed separately. Two fresh nine-round
processes plus an isolated eviction-control process, native O3 on one gfx90a
GCD, CPU+memory NUMA node 0, no model/context: tile-k versus the best confirmed
affine CK choice at M>=128 has warm graph-batch changes -4.9%..+3.6%, median
+1.0% across 36 distinct shapes (not a weighted model speedup). Examples:
512x14336x8192 ~77.8 -> 80.4 TFLOP/s; 4096x8192x8192 ~94.1 -> 96.5;
512x2048x8192 ~66.6 -> 63.4. Fixed-geometry comparisons and plain-versus-padded
controls remain distinct so tuning/padding does not masquerade as a format
benefit. N-major tiled storage consistently loses in this set. K-major storage
is therefore a viable shared GEMV/GEMM candidate, not universally superior;
affine and tiled CK still trail rocBLAS on many shapes. No production adoption,
model-level speedup, overlap or new hardware-saturation threshold is claimed.
The benchmark/GPU sources and measured executable are unchanged from the tiled
checkpoint; only drivers, analysis and findings are added. Full protocols and curated data are in the prefill research note.

**Compute-bound GEMM and effective-clock audit (2026-10, standalone):**
`research/gemm-compute-bound/` establishes an actual BF16-input/FP32-output
CK V3 256x256x64 GEMM at ~144.1 TFLOP/s (4096x6656x32768, no split-K),
native O3, one gfx90a GCD, CPU+memory NUMA node 0, no model/context. Its
~1385 MHz per-wave core/steady-counter rate differs from the constant-register
MFMA reference's ~1571 MHz /166.5 TFLOP/s. Normal signed reference operands
alone reduce that rate to ~1426 MHz /151.1 TFLOP/s. Clock-adjusted reference
~146.8 puts GEMM at ~98.2%; per-wave work is ~250 versus ~255 FLOPs/core-cycle.
Firmware average GFX corroborates this; current GFX remains 1700. No fixed
clock guarantee, specific power-policy cause or physical traffic is claimed.
Repeated affine/B-panel/AB-panel controls rank AB-panel > affine > B-panel
when each uses its better measured CTA order at all nine confirmed problems,
but B-panel has 16 private bytes/thread and a hot-loop spill reload versus
zero for the others. A longest-K fixed-order exception is retained. This is
compiled-implementation evidence, not a universal tiling rule or promotion
of the compute-sized panels over the earlier shared 32x128 weight format.
New BF16/FP16, INT8, mixed-load and SIMT GEMV clock probes reproduce the rate
effects with instruction/count checks. Mixed loads also reduce per-cycle
MFMA throughput; the GEMV probe excludes combine and has only structured-input
y[0] validation. Original/fresh logs, layout controls, code-object audit and
scope limitations are retained in the research notes. Follow-up hardware-counter
profiling is now usable with the existing local Compute/SDK/AQL installation.
Two 12-pass actual-GEMM captures and three same-pass core repeats per case
confirm long-K AB-panel at ~97.70% counter-derived MFMA utilization (emulated
busy-cycle formula), zero reported LDS bank conflicts and only ~147-163 GB/s
L2-to-HBM read payload for fixed n-fast. B-panel issues ~6.24% more VMEM
instructions; changing spill-free affine to m-fast restores utilization from
~86.88% to ~96.57%. Mixed-version profiler metadata/blank-field limitations,
raw count checks and profiling-versus-unprofiled timing distinctions are
recorded in the profiling note. No production, format, dispatch, precision-
contract or clock/power configuration changes are made.

**Unloaded memory-latency baseline (2026-10, user-directed):** standalone
`research/memory-latency/` measures one dependent wave on HIP 0 / PCI
0000:31:00.0, native O3, CPU+memory NUMA node 0, no model/context or clock
changes. Two nine-round random-cycle runs retain 1,134 samples across cache-
sized to 256 MiB allocations. For a coalesced 256 B wave load, median raw
single-load intervals are ~160 cycles at 4 KiB, ~296 at 64 KiB/4 MiB and
~616-708 at 64-256 MiB, including a measured 44-cycle empty interval. The
65,536-step dependent sequences include address/wait/loop overhead and reach
~411-469 ns per step at the large sizes, at ~1685 MHz light-load core rate.
ISA verifies load/wait/timestamp ordering and zero private/LDS storage; 63
counter records confirm exact load counts and the small-set default versus
glc L1-bypass behavior. Single-request variation, approximate timer subtraction,
translation/cache effects and the distinction from loaded GEMV are explicit.
No actual-GEMV latency-under-concurrency or unique DRAM/channel cause is claimed.
A fixed-payload loaded-window follow-up now sweeps 256/1024 B wave loads,
depths 1..16 and 4..416 CTAs on 1040 MiB. Two nine-round runs retain 1440
unprofiled samples; three native-SDK counter passes retain 240 cases with exact
wave/VMEM/payload checks. At 104 CTAs, wide depths 1/2/4 reach ~1.02/1.41/1.47
TB/s with nominal 416/832/1664 KiB windows. Native VmemLatency works, measuring
~0.51/0.97 us instruction lifetime at depths 2/4 and ~725/1436 KiB mean
payload-equivalent populations. Nominal 512 KiB configurations top out at
~1.165 TB/s; larger windows are not uniformly faster. These are shader-level
estimates (including fragment/completion and small store effects), not physical
HBM queue occupancy or independent proof of Little's law. Burst/drain grouping,
request distribution, deep-case variability and actual-GEMV scope remain explicit
in the window note. All entries are spill/LDS-free; no production changes.

**FP8 W8A16 bring-up (user-directed, standalone):**
`research/fp8-w8a16/` tests the user's local AMD matrix calculator/venv
and implements E4M3FN-to-FP16 expansion using two permutations, two packed
signed shifts, two masks and two packed FP16 x256 multiplies. Device checks
cover every finite byte encoding in every packed position, a dense 32x32x8
FP64 oracle, and all 14 signed FP8 subnormals restored before MFMA (the
unrestored control demonstrably flushes). A repeated-consumer test catches
an inline-asm scheduling hazard; the current register probe uses alternating
operand banks and next-conversion/current-MFMA ordering, without hot-loop NOPs.
Native O3, HIP 0 gfx90a, CPU+memory NUMA node 0, no model/attention context:
two nine-round processes reach 141.885/141.799 TFLOP/s versus matched FP16
149.467/149.402 (~95%), with ~66 versus ~64 core cycles/MFMA and lower
effective clocks. This is not yet full-speed or a general tiled GEMM.
Basic eight-load batch/drain controls reach ~46 FP8 versus ~36 FP16 TFLOP/s;
streamed FP8 still retains conservative producer/consumer separation, not a
rotating future-load pipeline. All checked throughput outputs match FP64
exactly in the retained runs; executable-object audit verifies instruction
counts and zero private/LDS bytes. NaNs/FNUZ, activation subnormals, scales,
LDS sharing and real-model integration remain outside this prototype. No
production or model-format changes. The user accepts the ~95% arithmetic
rate; a diagnostic NOP-window sweep (`research/fp8-w8a16/nop-sweep.md`)
confirms the predicted boundary-amortization behavior at 2/8/32 MFMAs per
backedge. Two nine-round processes, same native O3/GCD/NUMA placement and
no model/context, check all 135 variants per round with zero observed
FP64-reference error. For 0..24 added nominal core cycles/conversion, the
per-MFMA slope is 1/U; around 28 the knee appears, and by 32 the slope is
one-for-one. Independent scalar-only calibration measures ~4 core cycles
per encoded NOP wait-state unit. Exact-interval ISA/resource audits and an
8192-step endpoint control pass. This is scalar issue-gap tolerance, not
promised useful VALU/VMEM overlap; no production tuning or dispatch changes.
A user-proposed MFMA-first/conversion-after phase rotation is tested in
`research/fp8-w8a16/ordering.md`: two primed operands preserve separation,
and exact-ISA audits confirm the changed order with the same target wait
and register counts. Two nine-round 4096-step runs plus an 8192-step control,
same O3/GCD/NUMA/no-context protocol, fit ~20 versus ~12 cycles/backedge
(after versus before); all 243 samples match FP64 exactly. The bubble
persists and slightly worsens, so the accepted original ordering stays the
reference. The specific wait/branch/forwarding mechanism remains unproven.
The subsequent four-bank/staged-VALU study (`research/fp8-w8a16/dependency.md`)
finds no gap-depth fix: matched two/four-bank before costs ~12/~20 cycles
per backedge, after ~8/~24. Fully primed ring prologues change the emitted
loop wait to a compiler NOP, so those absolute after constants must not be
silently substituted for the earlier order probe. Raw-FP16 controls already
recover a bubble with seven vector NOPs or independent VALUs (~4 cycles per
backedge; eight -> ~8); barriers-only/tail counter placement does not suffice.
Source-write thresholds include explicitly audited compiler copies/NOPs.
Two nine-round 4096-step runs plus a three-round 8192 control, same native
O3/GCD/NUMA/no-context protocol, yield 6300 exact-valid samples; 189 unsafe
real-sign-flip/before samples fail and are excluded from fits. Physical bank
distances, source registers and zero private/LDS bytes are audited. No
production change or unique hardware-mechanism attribution is made.
The user's next preheader/split-conversion plan succeeds in the isolated
register prototype (`research/fp8-w8a16/splitting.md`). Explicit AGPR init
and load readiness before timing remove all loop-head waits/NOPs; primary
two-bank 8/8 retains ~4 cycles/backedge, then hand-enumerated 10/6 (also 9/7)
placement removes the fitted boundary term. Same native O3/GCD/NUMA/no-context
protocol: two nine-round 4096-step processes and a three-round 8192 control,
3528 samples, all exact against FP64. Primary two-bank 10/6 is ~64.009 core
cycles/MFMA versus raw ~64.010; event rates ~145 versus ~150 TFLOP/s
(96.54%/96.53%) retain a lower effective clock, so equal wall-time throughput
is not claimed. Exact interval/lifetime audits verify no extra copy VALUs,
no in-loop memory, zero private/LDS bytes, unchanged reported 28 VGPRs for
the primary controls/splits. Unfavorable pad/four-bank variants remain
recorded; neither a universal padding rule nor a unique hardware cause is
claimed. Future-weight streaming and general GEMM integration remain open;
original probes and production code are unchanged.
The register-only research checkpoint is committed as `8e1c120`. The next
user-directed phase adds real synthetic-weight global loads, initially with
one immediate `vmcnt(0)` drain per future fragment and no latency/backedge
tuning (`research/fp8-w8a16/loads.md`). Native O3, HIP 0, CPU+memory NUMA 0,
no model/context, U=8 at 4096 MFMAs/wave: two nine-round processes give
FP8 ~584.8/586.0 cycles/MFMA, 18.514/18.433 TFLOP/s and 289.274/288.017
useful weight GB/s; matched FP16 ~612.7/612.2 cycles, 17.630/17.667 TFLOP/s
and 550.928/552.085 useful GB/s. These are unhidden-wait baselines, not
bandwidth ceilings or physical-HBM counts. Footprints are ~416/832 MiB;
8192-step repeats agree. All 252 timed samples and 72 changed-input/poisoned-
output/guard validations match FP64 exactly; actual-ISA audits verify real
loads/full waits, no private/LDS storage or lane spills. The user requests
reporting before tackling the per-MFMA gap, then backedge tuning; that initial
baseline leaves both untouched. No production change is made.
The subsequent user-directed raw-load queue sweep (`research/fp8-w8a16/pipeline.md`)
keeps A freely register-available and tests D=1/2/4/8, with partial waits
`vmcnt(D-1)` and no full body drains. Same native O3/HIP0/CPU+memory NUMA0,
no model/context, U=8, 4096-step protocol: FP8 reaches 18.934/35.155/62.229/
86.308 TFLOP/s in run 1; D8 repeats at 85.794 (4.56x/4.55x versus matched
D1), ~1.35/~1.34 TB/s useful weight payload and ~117/~116 cycles/MFMA.
An 8192-step control agrees. All 420 timed samples and 120 changed-input/
poisoned-output/guard cases match FP64 exactly. Actual-ISA FIFO/register-
lifetime auditing verifies queue credits and rejects premature pending-value
copies; wait and first read are fused to avoid an initial opaque-asm hazard.
All entries remain private/LDS/lane-spill-free. Common padded layout keeps
addresses identical across depths; FP16 controls include explicitly counted
safe copies. No activation loading, operand reuse, grid or backedge tuning,
compute-bound claim or production change is introduced. Primary effective
shape is M=32,N=13312,K=32768: 416 waves each own a 32x32 output tile and
traverse full K, with the same 32x8 activation fragment repeated under the
free-A assumption. The 8192-step control doubles K. The user recognizes the
FP16 control's ~1.38 TB/s as the GCD's practical streaming ceiling (FP8 here
~1.34–1.35), and requests preserving the load/convert loop as a future FP8
W8A16 GEMV reference candidate, not a validated M=1 implementation.
The approved real-A round (`research/fp8-w8a16/paired.md`) materializes shared
MFMA-packed FP16 A[32,32768] (2 MiB plus lookahead) and uses A-first/B-second
load pairs with `vmcnt(2D-2)` at depths 1/2/4/8. Same native O3/HIP0/CPU+memory
NUMA0/no-context/U=8 protocol: real-A FP8 depth 8 reaches 80.793/80.812
TFLOP/s versus co-run free-A 84.865/85.411; core intervals remain ~117 cycles
in both, with lower real-A effective clocks. All 672 timed samples and 384
independent A-only/B-only/both-change/poison/guard validations are exact.
The 4 MiB A/K=65536 control reaches 81.541 TFLOP/s. All 32 code objects
pass paired-order/FIFO lifetime/resource audits; no private/LDS/lane spills.
A request rates (~2.525 TB/s at D8) count repeated wave reads to shared
addresses, not physical HBM traffic or a measured cache hit rate. No new
M/N reuse, grid or backedge tuning, engine integration or compute-bound
claim is made; this measurement round is concluded.
The next requested CTA-grid test (`research/fp8-w8a16/grid.md`) uses 13 N by
8 M blocks, eight distinct A streams, and four waves still entirely along N.
B is naturally shared across eight M groups. Effective shape changes from
32x13312x32768 to 256x1664x32768 at identical FLOPs/output count; active
logical A/FP8-B changes from 2/416 MiB to 16/52 MiB. Both grids launch the
same compiled entry per format/depth/unroll. Same native O3/HIP0/NUMA0,
no-context, D8/U8 protocol: FP8 104x1 -> 13x8 is 81.989/81.850 ->
91.060/92.179 TFLOP/s and ~115/~116 -> ~103/~102 cycles/MFMA; FP16
~41.5 -> ~68.2/~68.6 TFLOP/s. New aggregate A+B shader payload is ~4.3 TB/s
for both formats, not physical HBM bandwidth; unique-matrix equivalents
are recorded separately. All 672 timed samples and 384 independent changed-
input/poison/guard cases are exact; the larger-K control agrees. Actual-ISA
paired FIFO/resource audits pass for all 16 shared entries, no private/LDS
or lane spills. Shapes, reuse and footprints change together, so no unique
NoC mechanism is assigned. No 2x2 wave tiling or production change is made;
this round is concluded before the next wave-layout discussion.
The user then selects only a single-CTA latency-coverage route
(`research/fp8-w8a16/latency.md`), retaining four N waves and testing paired
D1/2/4/8/16, fixed unroll32, resident MFMA anchor and cache controls. An
in-dispatch pre-read replaces ineffective inter-dispatch warming; a completed
256 MiB eviction reader is outside probe timers. Native O3/HIP0/NUMA0,
no model/context: two nine-round 4096-step runs give raw ~64 cycles/MFMA,
FP8 D8 pre-read ~100 and eviction-controlled ~117; D16 ~104/~120–121 is
not an improvement. A three-round 1024-step cache-small control gives
~96 warm FP8 D8, ~134 FP16, still above the resident anchor. Runtime L2
is 8 MiB; working-set/counter fields and same-dispatch cache qualification
are explicit. All 528 timed and 352 independent changed-input/warm-cold/
poison/guard cases are exact; D16 paired FIFO and no-spill/LDS audits pass.
Event launch_ms includes the pre-read; only body clocks compare protocols.
No architectural minimum-latency or unique NoC/local-bandwidth cause is
claimed, and no 2x2 tiling or production change is pursued. This round is
concluded: buffering alone does not recover 64 in the tested implementation.
The correct fixed-shape real-A W8A16 GEMM/grid/diagnostic progress is committed
as `21c735b` before the user's intentionally non-GEMM fixed-K test
(`research/fp8-w8a16/fixed.md`). All A/B priming, queue-fill and refill
addresses freeze at the first K=8 fragment, but native loads/waits/conversion
remain. One CTA/four N waves, same O3/HIP0/NUMA0/no-context, unroll32:
fixed D8 FP8 reaches ~96–97 cycles/MFMA, FP16 ~129, versus resident ~64;
8192 repetitions give ~96/~128–129, including D16. Warm/eviction-prepared
results converge for the 1.5/2.5 KiB active input sets. Logical payload is
3072/4096 B per four-wave step, both near 32 B/core-cycle at saturation;
this suggests a local load/result-delivery constraint, not proof of a
specific cache/port or an observed hardware bandwidth counter. Address
arithmetic also folds away. All 1008 timed and 672 independent changed-input/
warm-path/poison/guard cases are exact; actual-ISA audits prove invariant
pointer operands, retained loads/conversion and FIFO safety with no spills/
LDS. No performance counters, 2x2 layout, corrective tuning or production
changes are introduced; this fixed-fragment round is concluded.
User then requests intentionally unsafe register reads and stops numerical
correctness testing: `research/fp8-w8a16/waits.md` removes all steady partial
waits, retaining identical instructions/registers otherwise and initial/final
completion. Deep FP8 D8 stays ~96 cycles/MFMA, FP16 ~129; D1 improves sharply.
No output readbacks or numerical tests run. A subsequent same-binary shallow-K
sweep (`shallow-k.md`, same native O3/HIP0/NUMA0/one CTA/no context) covers
32..4096 MFMA repetitions, logical K=256..32768. FP8 D8 no-wait follows
`T_body approximately 96*steps+460` core cycles: 110.594 cycles/MFMA at K256,
96.113 at K32768. FP16 falls ~248 to ~129. Deep K does amortize finite-length
costs, but not the steady 96-cycle slope. Other unwaited FP8 depths exhibit
~13–15k residual cycles above `96*steps`; inherited warm-up favors FP8 D8 and
the resident anchor, so these intercepts cannot be assigned to queue depth or
a particular hardware latency. Packed pitch/allocation also vary with length.
No extra warm-up policy, kernel/layout change or numerical tests accompany
this timing-only rerun.
After the user stops an idle but VRAM-resident llama.cpp server, an exact-binary
K256/K32768 timing-only recheck (`research/fp8-w8a16/server-off.md`) finds the
long-loop floor unchanged: FP8 D8 no-wait 96.113 -> 96.109 cycles/MFMA, FP16
128.563 -> 128.875 (same HIP0/native O3/NUMA0/one CTA/no probe model/context).
All long-loop case medians move by <0.87%; short-loop variation persists.
Before/after AMD-SMI snapshots show no processes and ~10 MiB/GCD; HIP0 is
AMD-SMI GPU2/BDF31:00.0. This does not exonerate every historical measurement,
but server residency does not explain the current ~96/~129 floor. Numerical
tests remain off and no kernel changes are made.
The next user-directed ablation (`research/fp8-w8a16/b-only.md`) keeps A in
one register pair and removes all timed/queued A loads and the two A-copy
VALUs, retaining fixed-address B loads/conversion at logical K32768. Same
native O3/HIP0/NUMA0/one CTA/four N waves/no probe context: no-wait D8 FP8
reaches 64.391/64.391 cycles/MFMA, FP16 64.653/64.661, versus fresh matched
A+B 96.114/96.109 and 128.965/128.939. Restored B waits at vmcnt(D-1) also
reach ~64.4/~64.9 at D8. Executable ISA confirms one B load/MFMA, invariant
unwritten A, eight retained FP8 conversion VALUs and zero A copies; no math
tests or output readbacks run. Four-wave logical payload falls from
3072/4096 B to 1024/2048 B, agreeing with the fitted 32 B/core-cycle plus
64-cycle MFMA-floor model. This does not separate A-load from A-copy costs,
identify a physical port, or claim the same rate for streaming B from HBM.
No production integration or other layout/clock/warm-up changes are made.
The user-directed MFMA-to-NOP hypothesis test (`research/fp8-w8a16/b-nop.md`)
then replaces each streaming MFMA with `s_nop 0`, preserving compiler operand
lifetimes. Actual-code-object inspection finds all remaining instructions
and registers identical to MFMA controls, except branch displacements.
Same native O3/HIP0/NUMA0/one CTA/four N waves/logical K32768/no probe context:
no-wait D8 FP8 is 45.394/45.394 cycles/step (not the payload-only prediction
32), FP16 64.601/64.592 (close to prediction 64). Restored waits give ~48.9/
~64.7. This is a partial match, not validation of the complete payload-only
model or a physical memory limit. FP8 conversion/NOP/issue/dependency work
remains and its contribution is not isolated. No further ablation, numerical
test or output readback is performed; no production changes.

The next user-directed framework (`research/fp16-load-matmul/`) is FP16-only
operand math with FP32 accumulation, one CTA/256 threads/four waves and one
AGPR accumulator bank/wave. It exposes hot, cyclic L2-sized and no-wrap
streaming modes; any depth 1..16 and unroll 1..64; coalesced 1xdwordx4,
2xdwordx2 or 4xdword A+B loading; and optional steady vmcnt waits. An opaque
asm generator protects pending registers; phase-specialized U-step bodies
support U<D and non-dividing D/U without pending copies, reporting code-size
expansion. It saves absolute wave clocks and uses CTA max(end)-min(begin),
not wave-median timing. Native O3/HIP0/NUMA0, no probe model/context, D8/U32,
logical K32768: two runs hot/no-wait give 64.133/64.133 cycles/step for 1x4,
128.234/128.234 for 2x2, 96.458/96.458 for 4x1, at identical 4096 B logical
payload/step. A 1 MiB primed cyclic window and ~16 MiB streaming window are
also measured; these labels are not counter-proven cache residency. D16/U1
and D3/U8 example timings and all 54 corresponding code entries are inspected
with no scratch/LDS/lane spills. No numerical/output tests or production
changes are introduced. Documentation includes the actual static phase count,
per-configuration warm-up, layouts, raw timings and reproducible commands.
The requested LDS extension (`research/fp16-load-matmul/LDS.md`) adds local
hot and 64 KiB sweeping modes, directly initializing LDS before timing (no
VRAM-to-LDS operand transfer). 32 banks x512 dwords, bank-balanced stride1
and deliberately concentrated stride4 controls; ds_read_b128/2xb64/4xb32 use
lgkmcnt, not vmcnt. Timer reads also use LGKM, so raw LDS priming is inside
timing after start-counter completion, not accidentally drained before the
loop. On the same native O3/HIP0/NUMA0/one CTA/K32768/U32/no probe context,
waited hot D1 is ~89/~89/~99 cycles/step; D2 reaches ~64.04/~64.03/~64.29.
Wide 64 KiB sweeps reach ~65 at D2; scalar sweeps retain non-monotonic
~66–69 overhead. D means raw slots PLUS two ready banks, not total buffers.
Stride4 uses eight banks rather than 32 and stays ~128 cycles even at D4.
LGKM thresholds are explicitly capped at 14 rather than the all-ones 15; clean
D1..4 minimum-depth findings are uncapped. Seven builds/210 entries are
inspected and 1512 timing samples saved, with no numerical/output tests.
Old v1 binaries and pre-extension source are preserved; no production change.

User then re-enables full-output checks (`research/fp16-load-matmul/VERIFICATION.md`).
The independent integer/dyadic host matmul reconstructs all 4096 MFMA outputs,
with poisoned buffers/32 guard words, legacy and hash inputs, independent A/B
seed changes and partial-phase exits. All 531 waited cases pass exactly,
including 96 direct original-code-object launches via HIP modules. A further 324
checked timing passes also pass; fresh cycles are slightly higher, e.g. LDS
D2 ~65–67 and global hot ~65/~129/~99 rather than a bit-identical timing claim.
8/12 deliberately unsafe no-wait cases fail, and observed matches are not
certified. Default verification is ON, checks outside timing; prior binaries
remain intact. No production change.

Next, the one-ready-bank first step (`SINGLE_READY.md`) deliberately RETAINS
all four raw->ready copies to isolate immediate copy->next-MFMA consumption.
Raw D and wait thresholds are unchanged; R1 seeds operand0 and prepares n+1,
R2 seeds0/1 and prepares n+2. v4..v7 remain reserved to control register layout.
All 150 R1 verification cases: hot 60 pass, varying 90 fail; all 135 R2 controls
pass. Warmed timings add 198 launches/variant: R1 sweep 72 fail, R2 all pass.
Hot timings remain near controls (~65 LDS D2, ~65/~129/~99 global D8), but
hot's repeated operand values mask the R1 correctness failure. Generated-asm
symbolic execution supplies the intended record/word sequence under ideal
completion; precise hardware hazard/required spacing is not isolated. No NOP
repair, copy-free path or throughput claim for invalid results is introduced.
Default R remains 2. Both variants and complete failing artifacts are retained.

User next requests removing ready storage and feeding MFMA from the counter-
completed raw entry (`DIRECT_RAW.md`). R0 primes raw0..D-2, issues n+D-1,
waits, and consumes raw[n%D] directly: zero ready banks/copy VALUs, no added
arrival stage or load->MFMA delay. 1224 positive checked launches pass exactly
(495 direct verification cases, 45 R2 controls, 684 checked timing passes).
Copied-R1 controls still fail3/3 with a common post-timestamp accumulator-
harvest guard, which never pads the measured loop. Same native O3/HIP0/
NUMA0/one CTA/K32768/U32/no probe context: R0 hot LDS b128 D1 ~69.55 cycles,
D2 ~64.58; sweep D1 ~82.8, D2 ~66.1. D2 now means two TOTAL operand slots,
not raw2+ready2. Fresh R2 hot D1 is ~90.4 and D2 ~65.0. Direct global hot D8
one/two/four loads are ~64.31/~128.5/~85.37, all verified; bank-stride4 still
~128.4. The current completion wait suffices in tested paths, without the
proposed extra arrival stage; exact internal decrement timing is not proved.
Twelve builds/360 entries inspected; earlier binaries remain untouched.
Default R2 stays compatible; R0 is explicitly selected. No production change.

Before culling, FP8/FP16 progress and complete samples are checkpointed in
`3986b13`, excluding unrelated working changes. First cleanup removes active
FP16 wait-free generation/dispatch and unsafe runner cases: two-parameter
kernel templates, 15 mandatory-wait entries rather than 30. `--wait 1` is a
compatibility alias; `--wait 0` and original no-wait module symbols are rejected
before explicit GPU initialization. Waited original-module replay is retained.
Seven builds yield 105/105 identical surviving timed instruction/register lists
after branch/address normalization; 223 exact cases plus 6 runner cases pass,
and 8 rejection checks pass. Results are in `results/waited-only/`; current
build paths append `-waited`. Historical raw data and FP8 diagnostics remain
preserved; copied-R1 negative paths are not part of this first cull. No engine,
CMake, model-fixture or production changes.

The next cull removes ready-bank generation/selection, copied-R1/R2 branches,
seed-ready/copy code and original-module replay from the active FP16 probe.
Five checkpoint-specific replay drivers are deleted; historical data/docs are
retained and code is recoverable from `e260a39`. Only direct raw consumption
with mandatory waits remains. Current matrix (`MATRIX.md`) is five address
modes x three load shapes = 15 kernels/build, independent D1..16/U1..64 and
LDS bank stride1/2/4/8; runtime windows/K/data/measurement knobs remain.
D is total operand slots, not raw plus ready. Build signature is four
arguments (D,U,optional output,stride), with `-direct` output names; old ready
selectors/module options are rejected. Seven representative builds retain
105/105 identical direct timed instruction/register lists; 222 exact cases
plus 6 matrix-runner cases and 15 removed-option rejection checks pass. The
constant-zero ready CSV column is compatibility metadata, not an axis. No
production changes or new performance campaign.

The next user-directed standalone study (`research/fp16-gemm64/`) validates
four waves in a 2x2 M/N grid, then implements actual one-CTA 64x64xK FP16
GEMM with FP32 output. Each staged A/B tile serves two waves: 2048 unique
bytes per K8 step, while nominal LDS-to-register requests remain4096 bytes.
Two producer waves stage A, two B; all four compute. Common MFMA-packed FP16
input is rearranged from row-major matrices OUTSIDE timing, with zero-padded
K tails; these are not end-to-end row-major-pointer GEMM timings. Batched
register staging is a non-overlapping control. Native gfx90a direct staging
uses global_load_dword ... lds, preserves m0, and overlaps future tile writes
with current computation in two LDS buffers. Explicit vmcnt completion and
whole-CTA barriers publish/reuse tiles; compute LDS reads use lgkmcnt.
Initial 2x2 sharing checks pass before GEMM addition. The final suite passes
620 exact full-C/guard launches, each GEMM case also checking both staging
buffers word-for-word. Six builds/48 entries have no scratch/lane spills or
inner accumulator copies; ISA confirms actual direct-to-LDS and no hidden
VMEM drain between future issue and current MFMA. Native O3/HIP0/NUMA0, one
CTA/256 threads, K32768, D2/U4, no model/context/clock changes: best tested
StageK128/direct/two-buffer runs are 347.360/346.240 us, 772.787/775.287
GFLOP/s per CTA, ~143 cycles/MFMA (not compute-bound ~64). Producer issue,
stage reset/synchronization and non-overlapped reg-control limitations are
explicit; no physical traffic, optimality or full-GPU claim. Round concluded
without production integration or additional optimization.
The completed 2x2/GEMM study is checkpointed at `3217640`, then the losing
register-staged producer is removed from active source/generation/dispatch
and runner cases. Direct-to-LDS becomes the sole producer; at that checkpoint
only one/two LDS buffers remain selectable. Five GPU entries/build (two GEMM, two LDS diagnostics,
one copy validator) replace eight. Six builds/30 entries inspected and two
removed-selector rejection checks pass; all 24 surviving benchmark core
instruction/register lists match after branch/address normalization. The
initial 340-launch runner was later found to name archived pre-cull binaries
(default direct path), not the pruned builds. Those raw runs are retained
and flagged, rather than claimed as current-binary numerical validation.
Cycles per four-wave MFMA step are now the primary report:
K32768 executes 4096 steps/wave, so metric=CTA core span/4096, including initial
staging, compute/waits and stage synchronization, not isolated MFMA latency.
`body_us` measures that region with the steady counter; `launch_us` is the
full event interval including epilogue, neither includes host packing/upload
or verification. Then-reported StageK128/two-buffer runs were 143.252/143.403
cycles/step (~347 us); these reported timings also came from the archived
binaries because of that runner path error. Explicit `-direct-only` paths
and pre-launch source/binary SHA checks fix it; rebuilt current binaries
subsequently pass 340 checks in `results/direct-only-corrected/`. Historical
reg results remain archived; no production/CMake/model changes.

The requested StageK8 /2,4,8-slot LDS study first reused a per-record inner
compute helper, giving ~709-712/~435/~433 cycles/step at K32768, D1/U1,
native O3 HIP0, CPU+memory NUMA0, one CTA/256 threads, no model/context/clock
changes. Its 405 checked launches are valid but the protocol resets raw
prefetch each record and synchronizes per K8; it is not a faithful continuous
register-pipeline test. Full source/binaries/results are preserved separately.
The user's correction makes two raw register buffers persistent across the
entire K loop, removes the inner compute call, and explicitly unrolls the
outer bulk by 2/4/8. Fixed DS offsets/raw parity, one source-pointer state,
one counter update/backedge per unrolled group. Prefetch next, issue current
MFMA, then wait for next LDS reads; retire that prefetched slot at the CTA
barrier before refilling it, while its operands survive in registers. Same-
class VMEM waits leave future copies in flight. Native ISA checks the bulk
instruction counts, alternating buffers, literal offsets, wait order and
absence of scratch/inner accumulator copies. No copied ready bank or payload
VGPR staging is reintroduced. Two randomized nine-sample runs at the same
placement/shape give 2/4/8 slots 642.498/642.400, 227.601/227.034, 165.684/165.899
core cycles per four-wave MFMA step; eight-slot body 403.200/403.520 us.
Matched StageK128/two-buffer control 143.217/142.946 (~347 us). The new raw
D2/U1 LDS-only hot reference is 64.183/64.189 cycles/step. Eight slots now help,
but do not beat the larger-chunk control; no unique hardware-cause claim.
All 405 checked launches pass, plus a fresh 340-launch rebuilt larger-chunk
regression; source/hash-bound records and retained initial experiment are
under `research/fp16-gemm64/`. No production/CMake/model changes.

The user-directed ring-size follow-up tests 16 slots before conditionally
trying 32. Native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads,
K32768, no model/context/clock changes: 16 slots/32 KiB improve 8's ~166 to
157.876/157.887 cycles per four-wave MFMA step, but still trail the matched
StageK128/two-buffer ~143 control, authorizing 32. Final matched 16/32 runs
are 157.934/157.854 versus 158.934/158.408 cycles/step, bodies 384.320/384.160
versus 386.720/385.600 us. 32 fits the full 64 KiB (zero static LDS), but is
slightly slower in both runs, so the campaign stops here. No allocation
failure occurred and no larger ring was run. 32-stage startup primes 31 slots
(62 VMEM instructions/wave), retires the oldest at vmcnt(60), then issues
slot 31; steady vmcnt(60) keeps the pending population <=62, never 64 or the
disabled threshold 63. ISA checks the split prefix, full 32-step unroll and
fixed offsets through 63488. Both trials pass 486 exact checked launches,
including full-word/all-slot checks and short-fill/tail/ring boundaries.
Sources/binaries/data are preserved separately in `results/ring16/` and
`results/ring32/`; the runner accepts explicit paths/stages and refuses a
nonempty output directory. No physical-unit attribution, significant-
regression claim, production change or further tuning is made.

The next user-directed change retires CURRENT LDS tile n, not prefetched
n+1: issue next's two DS reads, lgkmcnt(2), MFMA current, publish n+2 with
vmcnt(2*(N-3)), CTA barrier, then refill current's slot with n+N. One fewer
global-copy stage leads the consumer; physical LDS/raw-buffer counts stay
unchanged. The next read pair may remain pending through the boundary.
No initial early refill/drain remains, and only the last tail record needs
lgkmcnt(0). N2 needs a separate post-refill publication barrier; N>=4 retain
one barrier/step. ISA checks partial-wait-before-MFMA, no bulk DS drain,
current-slot offsets, 26/58 VM watermarks, unroll/parity and zero spills.
Matched native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads,
K32768, no model/context/clock changes: 16/32-slot new runs 109.474/109.449
and 110.734/110.989 cycles per four-wave MFMA step, versus frozen old
157.906/157.862 and 158.195/158.298. New bodies 266.560/266.400 and 269.600/
270.080 us beat StageK128/two-buffer 143.130/143.483 cycles (~347-348 us).
All 882 checked launches pass: 720 new verification cases across 2/4/8/16/32,
plus 162 timed samples including matched controls. The harness timed out
late after all verification; hash/argument-bound recovery reuses 732 complete
processes and executes six remaining, preserving the interrupted pair and
original/recovery scripts, invocations and logs. Records/source snapshots:
`results/retire-current/`; current schedule described in CURRENT_RETIREMENT.md.
No old-retirement branch in the active binary, production changes, extra
optimization or unique hardware-bottleneck claim.

The user next requests ONLY the original post-MFMA read idea: for N>=4,
prime raw0/raw1 and publish tile2 at startup; then lgkmcnt(2), MFMA n,
immediately DS-read n+2 into that just-consumed buffer, publish n+3 at the
existing boundary with vmcnt(2*(N-4)), retire current and refill n+N. Thus
16/32 use VM24/56, with another global lookahead stage sacrificed but no
extra barrier, M0 optimization or operand padding. N2 remains its existing
special case. Fixed offsets/parity/unroll and conservative bulk bounds stay
unchanged. Native ISA verifies immediate same-buffer DS instructions after
MFMA, LGKM2 before it, VM24/56, no spills/inner loop/conversions. All882
checked launches pass (720 verification cases across all rings plus162 timed
samples including preserved baseline/controls), with no timeout this run.
Matched native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads,
K32768, no model/context/clock changes:16/32 candidate99.271/99.221 and
100.663/100.463 cycles/step improve pre-read109.625/109.694 and110.772/
110.878; candidate bodies241.600/241.440 and244.960/244.640 us. BUT8 slots
regress to153.702/152.277 versus137.669/137.879 in both repeats. This mixed
result is reported and performance work stops for the user's decision; no
alternate publication route, fallback or per-size selection is tried.
StageK128/two-buffer control143.302/143.235. Candidate assembly/data/source:
`results/post-mfma-read/`, baseline preserved at63d14a0. This tests the
actual asynchronous MFMA->DS->writeback sequence, not arbitrary immediate
VALU overwrites or a universal source-register lifetime. No production change.

The next user-directed narrow probe moves bulk `s_add_u32 m0` immediately
before the existing barrier and removes its explicit `s_nop 0`; startup and
bounded tails stay unchanged. Native bulk now has add/barrier/global-load
adjacency and no NOP,12 rather than13 instructions per wave/step; data ops,
waits, unroll, M0 restoration and the post-MFMA read algorithm are unchanged.
Native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768, no
model/context/clock changes: matched16 slots99.338/99.248 ->96.311/96.273
cycles per four-wave MFMA step, bodies234.400/234.240 us;32 slots100.706/
100.577 ->97.574/97.834.8 slots change direction between repeats, with no
reliable gain claimed. The ~3-cycle gain at16/32 indicates a visible gap/
setup scheduling cost but does not uniquely identify scalar issue, M0 or
barrier latency/a sole bottleneck. All882 checked launches pass (720
verification cases plus162 timing/control samples), with exact C/guards and
all LDS words/guards. ISA/frozen source/native fingerprints are checked;
zero scratch/lane/accumulator spills. Data/assembly: `results/m0-pre-barrier/`,
baseline61904f8 preserved. No further controls, production or system changes.

The user's next paired-copy optimization uses global displacements0/256
then2048/2304, advancing source4096 only after odd records. Since immediate
shifts both global and LDS addresses, M0 selects the EVEN slot's base for
both copies; effective destinations still match each current slot. Applied
to startup, split31/32 priming, bulk and odd tails; pre-barrier M0 setup and
all waits/data ops remain. Native checks half as many add/carry pairs,
paired immediates, compensated/effective M0 offsets and zero bulk NOPs.
Native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768,
no model/context/clock changes:16 slots96.800/96.611 ->93.510/92.646 cycles
per four-wave MFMA step (bodies227.680/225.760 us);32 slots97.273/97.943 ->
97.312/95.125 (one flat, one better);8 slots145.805/153.218 ->153.557/
154.193 (slower both, notable baseline variation). Results are mixed, not a
universal speedup. All882 checked launches plus20 targeted odd-fill/tail
smokes pass; all C/guards and separate all-slot words/guards exact. No spills
or dtype conversions; N>=4 bulk average12->11 instructions per wave/step. Source,
assembly and data `results/paired-global/`, deb88e1 baseline preserved.
No further tuning, physical-unit attribution or production/system changes.

The user-directed result scan/cull retains only the measured K8/raw2/16-slot
paired-global winner.8 is clearly slower;32 loses both latest repeats with
2x LDS;2/4-slot experiments are stale (not claimed as fresh paired timings),
and larger-chunk GEMMs/old raw-depth builds are superseded. Remove these from
source generation, dispatch, dimension/build flags and default runners; retain
all historical records, binaries and b081cef/source snapshots. Active code
has one GEMM, two opt-in LDS diagnostics and one copy validator: four entries
rather than eight, no legacy GEMM or two-slot/32-slot special paths. Fixed
constants are measured choices, not hardware constraints. The surviving
1652 timed instruction/register lines are identical after address/branch
normalization, so no algorithmic/pruning speedup claim. Default runner now
uses24 focused correctness cases +18 timing samples =42 checked samples,
not720+162; no archived baseline, larger-chunk control or LDS timing replay
by default. Full144 winner-only verification and four diagnostic checks are
opt-in. The42 default samples, four diagnostic smokes and nine removed-
selector rejection controls pass. Native O3 HIP0 CPU+memory NUMA0, one CTA,
K32768 cull recheck93.280/93.116 cycles/step (227.200/226.720 us). Selection
rationale and all scanned timing rows are recorded in MATRIX.md and
`results/active-cull/`. No production/CMake/model or unrelated changes.

User-authorized performance-only ablations separately patch copies of the
same verified native executable:16 steady MFMA sites ->one equal-width
v_pk_fma_f32 each (same distinct A/B VGPR reads, not AGPR-C/dependency
simulation), or16 VM24 waits ->s_nop0, or16 barriers ->s_nop0. One class
changed per variant, all other code-object/host bytes and metadata/resources
identical, same instruction widths/PCs; startup and32-record tail unchanged.
Native disassembly/re-extraction checks the actual embedded patched objects.
Original active binary remains unchanged. Native O3 gfx90a HIP0, CPU+memory
NUMA0, one CTA/256 threads, K32768, no model/context/clock overrides, two
nine-sample warmed process repeats: baseline93.325/93.043, MFMA-proxy87.074/
87.163, VM-wait-NOP93.427/93.229, barrier-NOP79.573/79.086 core cycles per
logical four-wave K8 step. Core/steady ratios remain ~1682 MHz. The largest
signal is the barrier/rendezvous and wave-interleaving constraints; VM-wait
removal has no net gain and MFMA proxy saves ~6 cycles, not intrinsic MFMA
latency. Effects are nonadditive; NOP cost and remaining gates can mask a
removed wait. No unique memory/issue/barrier hardware attribution. All72
TIMING samples are numerically UNVERIFIED (`--verify0`, valid=-1); only one
separate unmodified-baseline check is verified. These unsafe variants are
not valid algorithms or active search combinations. Manifest/ISA/raw clocks
and sources are in `results/ablations/`, analysis in ABLATIONS.md. No combined
ablation, further tuning or production/system change.

The user keeps64x64 M/N for CTA supply in real prefill and requests StageK16,
primarily preserving32 KiB (8 slots), also authorizing16 slots/64 KiB. Two
unchanged K8 records/stage, b64 reads and raw2 retained; no b128/data-layout
change. Each stage performs two MFMAs but only one VM wait/rendezvous/M0
setup-restore. Source still advances4096 per two fragments. Startup publishes
current/next stages; VM20/52 and a15-stage split-prime for64 KiB account for
four global DWORD instructions/stage without64 pending VMEM operations.
The last stage's second MFMA waits LGKM0; tails zero-pad to16 and reference
uses actual K. Native code checks two MFMA/four b64 reads per barrier,
fixed offsets/parity, M0/barrier adjacency, zero spills/conversions and8/16
slot metadata. No broad matrix or older algorithms are reintroduced.
Native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768,
no model/context/clock changes:32 KiB candidate92.821/94.006 cycles per
K8-equivalent MFMA step (226.080/228.960 us) versus frozen K8 baseline
93.452/92.976: no clear equal-budget gain.64 KiB92.241/91.959 (224.640/
223.840 us) versus93.271/93.276 is a small gain at doubled LDS, not a real-
prefill/CTA-residency recommendation. Both matched trials pass120 checked
samples total plus18 short-fill/tail smokes; all C/guards and staging words
exact. Baseline active/0528d65 source and binaries remain untouched. Data,
native/source hashes and assembly are in `results/stage16-s8/` and
`results/stage16-s16/`; analysis in STAGE16.md. No further tuning or production,
CMake/model/system/clock-policy change; b128 explicitly deferred.

New user-directed `research/lds-staging/` isolates pure global-to-LDS loading:
fixed1 MiB warmed source, four disjoint256 KiB wave ranges, fully coalesced
DWORD loads; destinations shared16 KiB or private16 KiB/wave. Both allocate
64 KiB and use identical source addresses. Opaque volatile asm issues64 loads
per81-instruction loop, zero loop waits/barriers/consumers/NOPs; terminal VM0
outside issue interval precedes completed timestamp and all-LDS check.
Timer LGKM waits stay outside loop. Requested payload per four-wave step
is1024 B, denominator loads/wave; absolute CTA timestamp makespans retained,
not per-wave duration max. Lane-periodic bits make overlapping/reordered
writes same-valued; all active/unused LDS and32 guards checked, not a source-
iteration oracle. Native O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA/
256 threads, no model/context/clock overrides: shared17.3966/17.3959 and
private17.3962/17.3953 core cycles/four-wave load,58.8621/58.8643 and58.8636/
58.8664 requested B/core-cycle (~99 GB/s); two counterbalanced warmed nine-
sample process repeats.36 timings plus6 traversal/wrap smokes pass. Source
fits intended L2-hot footprint but no counters certify residency. Layouts
are indistinguishable; pure-copy service reference, not proof of a specific
outstanding queue/count or universal physical bandwidth ceiling. Mixed LDS
operand-read/staging contention remains unmeasured. Source/native hashes,
ISA/metadata, clocks and commands in `research/lds-staging/results/base/`;
no existing research/production/CMake/model/system changes or further tuning.

Next user-directed LDS test separately measures one coalesced load+VM0 with
core-clock endpoints,64 samples/wave retained in distinct VGPRs. No record
stores/spills/consumers/barriers among samples; post-batch join prevents any
wave's output stores overlapping another's measurement.64 sequential kernel
batches/process,4096 intervals/wave; four256 KiB source ranges cover1 MiB
repeatedly after pretraversal/200 ms warmup, same shared/private destinations
and64 KiB allocation. Core clock-read/retirement overhead INCLUDED, no
calibration/subtraction/filtering. Native ISA checks exact sample sequences.
Native O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA/256 threads, no
model/context/clock changes: shared runs min232/236,max1456/2192,median296/
296,mean362.34/362.71,populationSD134.59/136.54; private min236/236,max1704/
1800,median296/296,mean365.39/361.80,SD137.44/137.63 cycles. P95~604–612,
IQR102–109, broad right tail; no systematic shared/private central gap.
All maxima in first measured batch are retained, not assigned a hardware
cause.65536 measured +512 smoke intervals, all record/check guards and
final LDS contents pass; periodic payload not a source-iteration oracle.
Raw/source/native records, per-wave/position stats in `lds-staging/results/
latency/`, analysis LATENCY.md. Separate binary preserves no-wait baseline;
throughput-period/serialized-latency ratio is not an exact queue-depth proof.
No further experiment/production/system changes.

**LDS disjoint-source CTA sweep (user-directed, standalone):** separate grid
probe preserves64 register-retained load/VM0 intervals/wave, disjoint1 MiB
source/CTA,1/8/16/32/64/128/256 CTAs. HIP0 reports104 CUs/8 MiB L2;
256 CTAs touch16 MiB/launch, above cache capacity, and grids above104 queue
because each CTA allocates64 KiB LDS. Native O3 gfx90a HIP0/BDF31:00.0,
CPU+memory NUMA0, block256, no model/context/clock changes. Main64-batch
source-revisit protocol captures33,095,680 unfiltered intervals. Shared
medians run1/run2 by ascending grid:292/296,580/580,584/584,592/592,684/684,
972/972,952/952 core cycles; private296/296,580/580,584/584,592/592,684/684,
972/972,952/952. All final per-CTA LDS and record/status guards pass. Local
post-sampling LDS checks avoid64 KiB global checker writes/CTA; small
record/status epilogues can still overlap other CTAs. Initial16-batch pilot
(8,273,920 intervals) preserved separately. Frozen-binary replay identifies
first source traversal batches0-15 at~580-590 cycles versus later revisits
~276-288, explaining the pilot's~590 versus original~296 median. Warmup
before H2D record initialization did not certify hot capture; cache warming/
capacity supported, exact prefetcher/coherency cause not isolated. Main data,
pilot and controls in `lds-staging/results/grid-latency{,64}/`, details in
GRID_LATENCY.md. No production/system or other research changes.

**LDS per-load VM watermark sweep (user-directed, standalone):** return to
streaming bandwidth at1/8 CTAs, private16 KiB/wave,1 MiB disjoint source/CTA.
Insert load then vmcnt(N), test0..32,40/48/56/62, disabled63 and no-wait.
Peak architectural instruction window bounded byN+1 (63 disables, not64 cap).
All waited variants patch only64 equal-width wait immediates in one verified
native template; identical host/other code, PCs/registers/metadata. Actual
embedded objects, disassembly, byte bounds and prelaunch hashes validated.
Native O3 gfx90a HIP0 CPU+memory NUMA0 block256, no model/context/clock
changes, two randomized/reversed nine-sample process repeats,1 GiB requested
per CTA/launch.1 CTA vmcnt15 gives18.203/18.545 cycles per four-wave load,
vmcnt16 reaches17.399/17.391 versus disabled63=17.396/17.395; a17-load
per-wave bound suffices.8 CTAs vmcnt16=39.287/40.795, vmcnt19=35.648/35.723,
vmcnt20=34.862/35.114 versus disabled63=35.429/35.383: broad noisy knee at
~20-21 loads, not the same threshold. Higher-window nonmonotonic points and
no-wait34.555/36.975 retained. No exact stall-free/hardware queue-depth claim;
loaded latency and throughput coupled. Core spans computed per CTA (median
CTA each round), global steady clock used for aggregate GB/s; all8 CTA bodies
overlap.1404 grid timings/6318 CTA records pass every LDS word/guard. Full
curve/raw clocks/source/native manifests in `lds-staging/results/windows/`,
analysis WINDOWS.md. Earlier measured binaries/sources unchanged, no further
schedule tuning or production/CMake/model/system/clock-policy changes.

**Global DWORDx4 ->VGPR ring (user-directed, standalone):** one CTA with
1/2/4 waves,32 slots x4 VGPR/lane, fixed1 MiB source split into disjoint
wave regions. One opaque asm continuously overwrites the ring, zero periodic
waits/consumers/copies/barriers, final VM0 outside issue interval. Native
loop53 instructions (32 loads+16 address+5 control),128 distinct payload
registers, metadata136 VGPR/24 SGPR, zero LDS/scratch/lane spills. Periodic
lane-varying payload makes final same-valued overwritten slots checkable,
not an arbitrary-data/last-writer oracle. All final slots/lanes/DWORDs and
32 guards exact:54 timings plus6 fill/source-wrap smokes pass. Native HIP
O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0, one CTA64/128/256 threads,
no model/context/clock changes, two counterbalanced warmed nine-sample
process medians. Default iteration counts adjust to1 GiB requested/CTA:
1-wave16.0226/16.0236,2-wave32.0446/32.0446,4-wave64.0783/64.0788 core
cycles per all-wave load step. Payload/step1/2/4 KiB, aggregate63.9097/
63.9058,63.9108/63.9110,63.9218/63.9213 requested B/core-cycle (~107.5 GB/s)
respectively. More waves do not increase aggregate rate in this configuration;
not a universal64 B/cycle hardware roofline or attribution to one return/
cache/register resource. Fixed32-slot depth not claimed optimal. Earlier
LDS58.9 comparison changes width/destination, not an isolated LDS-write cost.
Data/source/native manifests in `lds-staging/results/vgpr-ring/`, analysis
VGPR_RING.md. Existing measured sources/binaries unchanged; no further tuning
or production/CMake/model/system/clock-policy changes.

**VGPR ring coarse CTA knees (user-directed):** unchanged DWORDx4/ring32
no-wait asm,1/2/4 waves per CTA, grids1/2/4/8/16/32/64/104/208, disjoint1 MiB
source/CTA. Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0, no model/
context/clock changes. Proportionate survey uses two counterbalanced process
repeats x3 checked samples,100 ms warmup,256 MiB requested/CTA/sample. Source
footprint and concurrent CTA count coupled; not a cache-isolated experiment.
Near-linear1-2 CTA scaling, first sublinear/flat region around4-8, renewed
scaling16-32, broad best region32-64.104-208 regress, especially4-wave CTAs;
1/2 waves hold up better. Approximate peak region1.25-1.45 requested TB/s,
4-wave104/208 fall to~0.75-0.86, not a monotonic saturation curve. No attempt
to fix the regression or claim a unique cache/port/scheduling cause. All CTA
issue intervals overlap; HIP residency query12/6/3 CTAs/CU and104 CUs means
all grids within reported capacity, so not forced oversize-grid admission.
Native136 VGPR/28 SGPR, zero LDS/scratch/spills, same53-instruction loop.
162 timing launches/7902 CTA records plus3 grid-fill smokes pass every final
ring DWORD/guard. Global steady-clock grid makespan supplies bandwidth,
per-CTA core timestamps never mixed across CUs; full-event rate agrees
closely. Measured core ratios shift only modestly, not enough alone to
explain4-wave regression. Native/source/raw clocks/occupancy in
`lds-staging/results/grid-vgpr/`, analysis GRID_VGPR.md. No further tuning,
production/CMake/model/system/clock-policy or earlier measured-source change.

**64-CTA8x8 outer-product source sharing (user-confirmed):** same x4 VGPR
ring32/no-wait loop, four waves/CTA. CTA(i,j) waves0/1 read A_i512KiB,
waves2/3 B_j512KiB; eight row A/eight column B panels. Reachable source8MiB
versus disjoint64MiB, each CTA still traverses1MiB. Both allocate64MiB for
control; source address setup only layout difference in same executable.
GPU records256 source bases outside timing, all checked:32 distinct256KiB
regions used8x outer versus256 unique disjoint regions. Native138VGPR/30SGPR,
no LDS/scratch/spills, unchanged53-instruction generated timed loop. Native
O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 grid64/block256, no model/context/
clock overrides. Two counterbalanced nine-sample process repeats,200ms
warmup,256MiB requested/CTA/16GiB grid. Disjoint309.1617/309.2876 core
cycles per four-wave4KiB step,1376.75/1375.78 requested GB/s; outer139.3126/
138.0228,3013.17/3036.34 GB/s. Requested throughput2.19x/2.21x higher, not
physical HBM bandwidth or a real-GEMM speedup. Core ratios~1633 versus1604MHz
measured, not overridden.8MiB at nominal L2 capacity does not prove residency
or identify cache/miss-coalescing machinery.36 timings/2304 CTA records plus
4 fill/wrap grid smokes pass all final ring words/guards and source bases;
all CTA issue intervals overlap. Source/native/raw clocks/address maps in
`lds-staging/results/broadcast-vgpr/`, details BROADCAST_VGPR.md. Earlier
sources/binaries untouched, no further ring/wait/width/skew tuning or
production/CMake/model/system/clock-policy changes.

**Shared-panel rectangular grids (user-directed):**4x4/4x8/13x8 plus fresh8x8
control, four waves/CTA,512KiB panels and unchanged32-slot x4 no-wait loop.
Reachable shared pools4/6/8/10.5MiB for16/32/64/104 CTAs; each shape matched
against disjoint source/identical allocation. GPU source maps/reuse checked:
A reused by column count,B by row count (13x8 ->8x/13x). Native139VGPR/25SGPR,
zero LDS/scratch/spills, same53-instruction timed loop, runtime residency3
CTAs/CU; dynamic row/col setup before timing, one binary for all cases.
Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 block256, no model/
context/clock changes; two counterbalanced nine-sample process repetitions,
200ms warmup,256MiB requested/CTA. Shared cycles per four-wave4KiB step:
4x4=68.663/66.112,4x8=104.518/104.830,8x8=136.972/137.673,13x8=153.083/
151.863. Requested grid GB/s1553.72/1584.44,1972.34/1974.23,3064.68/3054.74,
4348.38/4383.01 respectively. Matched disjoint cycles146.470/151.158,
169.360/169.936,310.559/307.606,910.308/928.415; throughput ratios~2.13-2.24x,
1.57x,2.21-2.24x,5.71-5.86x.104-CTA sharing remains strong beyond nominal8MiB
L2 footprint, no capacity cliff here; temporal reuse/miss coalescing plausible,
not counter-isolated. Rates are requested/cache-reused, not physical HBM or
actual GEMM speedup; source footprint/concurrency/reuse change together.
144 timings/7776 CTA records +16 fill/wrap grid smokes pass final ring words/
guards/GPU source maps; all CTA issue intervals overlap. Native/source/raw
clocks/maps in `lds-staging/results/broadcast-shapes/`, analysis
BROADCAST_SHAPES.md. Earlier measurements untouched; no further producer/
wait/ring/skew tuning or production/CMake/model/system/clock-policy changes.

**Native full-GCD FP16 GEMM (user-directed, no BLAS/CK):** new
`research/native-gemm/` implements128/256 square output tiles,four waves/CTA,
FP16 packed A/B,FP32 row-major C,alpha1/beta0,no split-K/atomics. Selected
128x128x32 has wave64x64 register tile/four independent32x32 MFMA accumulators,
64 AGPR values/lane; requested global intensity64 FLOP/B,LDS-read intensity32,
both2x the prior64x64 scheme. One16KiB LDS buffer, wide global-to-VGPR and
b128 LDS ops, next-global prefetch, rolling next-fragment reads behind current
matrix work. Five raw fragment slots/two-phase q3 parity avoid operand copies.
Retirement barrier before overwrite and publication after completed writes;
two barriers/K32. Native hot loop(two stages)/wave32MFMA/8global-x4/8DS-write128/
16DS-read128, no conversion/accumulator movement/NOP/spills inside. HSA reports
136VGPR,64AGPR,46SGPR (not blindly summed); HIP query3 CTAs/CU. Native128
observed312 overlapping CTA bodies versus104 for256; occupancy/reuse/layout
changes coupled, not an occupancy-only causal proof.
Native HIP O3 C++20 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0,no model/context/
clock overrides,CPU packing/H2D/reference excluded. Full-kernel HIP-event
useful TFLOP/s, two counterbalanced warmed nine-sample process medians:
128 at1664x1024x8192=113.15/113.23,3328x2048x8192=137.66/137.66,
4096x6656x8192=141.58/142.07,4096x6656x32768=144.07/144.17. Corresponding256
34.46/34.42,122.66/122.68,124.59/124.92,128.37/128.35, so128 selected.
Largest128 effective core ratio~1418MHz,whole-GCD clock-adjusted efficiency
95.4/95.5%; per-CTA share~32.6% with3-way residency is not GPU utilization.
Near-peak in these conditions, not universal optimality or model-prefill claim.
Pipeline history preserved: initiallate K32~106/118TF; source-pair-bank and
LDS-XOR controls null; verified resident-first-fragment control body~98-99%
(not ordinary GEMM); early publication/rolling reads improve~113/123TF. K64
fits32/64KiB without spills but regresses~105/113TF, so keepK32/no more tuning.
Final180 verified launches:36 functional/input-seed plus144 timing. Small
full references; large~2-3K sampled exact original-row-major CPU dots plus
all-C finite-after-poison/guards, no library reference; do not claim full
large references. Exactness limited to integer/16 inputs,K<=65536. Input
packing is an explicit kernel contract/outside timing,not optimized end-to-end
row-major or production activation conversion. Source/native manifests, ISA,
raw per-wave binary clocks, overlap proof and dependency listings under
`native-gemm/results/`; all historicalsource/native hashes pass (Stage64
source recovered after host-only edits and SHA-matched, not relabeled).
README.md/DESIGN_LOG.md document design,resources,trials,limits/replay. Existing
research/binaries preserved; no production/CMake/model/session/system/package/
clock-policy changes. Research-log relocation remains user's working copy.

**Native GEMM readability/process audit (user-directed):** active code only
expanded into named host helpers and explicit device/packing steps. Generator
now has Configuration/PipelineEmitter methods for startup,retirement,
publication,rolling reads,tails and resident control. Logical TestMatrices,
FP16 encoding,packing and independent reference are readable separately;
benchmark/options/verification/clocks/reporting separated. Python runners,
native inspector,artifact verifier,bank control and shell build reformatted/
structured; critical Python validation is explicit rather than removable
assertions. AUDIT.md maps code,physical registers,barrier invariants,metrics,
trial provenance,limits and replay. Generated wrapper text deliberately kept
stable: all8 tile/stage/resident headers byte-identical. CPU equivalence
checks18 original/new matrix/seed pairs and144 complete packed vectors,
FP16 encodings and original-row-major dots. Selected128 and control256 GPU
.text sections byte-identical (7872/25664 bytes); bank control2372 bytes also
identical and8 control checks pass. All historical snapshots/data/binaries
untouched; rebased archived hashes,including bank control,pass. Analysis of
original data reproduces original reports byte-for-byte into a separate dir.
Fresh default artifact namespace `temp/native-gemm/readable/` preserves old
measured paths; source/CSV/clock format protocol unchanged. Rerun same180-
launch suite passes,selected long case144.069/144.024TF (~95.4% clock-adjusted)
versus original144.067/144.171: no algorithm/cleanup speedup claim. Proof in
`native-gemm/results/cleanup-audit/`,fresh results in `results/readable/`.
No producer scheduling/tile/math/packing/numerical policy,production/CMake/
model/session/system/package/clock-policy change. Scope is current native
GEMM code; archival source remains original rather than silently reformatted.

**Native GEMM208-CTA follow-up (user question):** previous suites had104/416/
1664 CTAs, not208. Unchanged readable128x128x32 binary now measured104/208/312
with N1024,M1664/3328/4992, plus208 aspect control M1664/N2048. Native HIP O3
gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0, no model/context/clock changes,
packing/reference excluded, two counterbalanced nine-sample eager processes,
K8192/32768.104 CTAs113.01/113.15 and119.73/119.80 TFLOP/s;208(26x8)
127.69/127.73 and132.52/132.59;208(13x16)127.78/128.01 and132.61/132.73;
312(39x8)134.91/134.88 and138.45/138.45.208 long-K global clock-adjusted
efficiency~86.5-86.6%,104~76.6-76.7%,312~91.3%. All144 grid timings pass
sampled exact references/all-C finite-after-poison/guards and raw-clock
validation. Fixed representative launch overlap reaches104/208/312 matching
grid counts. Resource cap remains3 CTAs/CU;208 does not enforce two on every
CU, no CU IDs/placement proof. Shape,reuse and admission behavior also change;
not occupancy-only causality. Data/commands/shared-driver snapshot and overlap
proof in `native-gemm/results/grid-coverage/`, notes GRID_COVERAGE.md. No
kernel/algorithm/packing/numerical policy or production/system changes.

**Native GEMM branch-only cleanup (user-directed,312 CTAs retained):** peel
last1/2/3 stages and guard two-stage steady entry with remaining>=4. Two full
phases always have valid stage+2 prefetches, subtract2/compare/backedge once;
no per-phase final/future checks or unconditional phase jumps. Current128
hot loop90->81 instructions/control12->3,256222->213; same32MFMA/8global-x4/
8DSwrite/16DSread/four barriers/six waits per selected two-stage backedge.
No fall-through s_branch0 in timed body. Same136VGPR/64AGPR/46SGPR,16KiB LDS,
zero private bytes; peeled-tail code grows selected .text7872->8704 bytes.
Memory/math/pointer/wait/barrier order unchanged: generated non-control traces
match at29 stage counts for all four tile/StageK configs; resident headers
byte-identical. Native loop non-control order also identical at128/256.
Correctness312 launches(60 bounded-tail/seed +252 timings),plus10StageK64
boundary checks,all references/finite-after-poison/guards pass. Large refs
remain sampled. Source/native snapshots and per-process executable hashes
record the actual before/after launches; no old artifact relabeling.
Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0, block256,no model/
context/clock override,packed operands/reference excluded. Two counterbalanced
nine-sample process repeats,104/208/312 CTAs atK8192/32768 plus large check:
208 long before132.611/132.560 ->135.258/135.323TF,global clock-adjusted
86.54%->88.53%;312 long138.478/138.451 ->139.838/140.025;104 long119.816/
119.857 ->122.487/122.465;large1664CTA143.985/144.013 ->145.919/145.879
(~96.7%). Modest positive control-flow result,not per-branch latency proof.
No MFMA/prefetch interleaving change yet;208 remains below90% target. Data,
source/native fingerprints and comparison in `native-gemm/results/branch-clean/`,
trace proofs in `results/branch-flow-audit/`,BRANCH_CLEANUP.md explains bounds/
parity and static-vs-executed counts. New build namespacebranch-clean preserves
readable/before binaries. Historical cleanup audit pins archived readable
source/builds so later intentional optimization does not invalidate it.
Default full runner retains208/312;focused A/B is separate. No producer,
packing,math,numerical-policy,production/CMake/model/system/clock-policy change.

**One MFMA after head VM0 (user-proposed):** move existing current-q0/C00
MFMA immediately after final head VM0 and before next-panel LDS stores;
remaining3 q0 MFMAs stay after staging/global-prefetch batch. No other
instruction reordering,control/tile/buffer/count/resource change. Selected
loop81 instructions,136VGPR/64AGPR/46SGPR,16KiB LDS/private0 unchanged.
Exact-move audit covers29 stage counts x4 normal configs: only stages-1
MFMAs moved, memory/wait/barrier/pointer and per-accumulator math order
otherwise identical; resident headers byte-identical. Actual128/256 native
hot loops match the exact two-move transform.312 checked launches(60
functional+252timing) plus10StageK64 boundary checks pass sampled/full refs,
all-C finite/poison/guards and raw-clock validation.104/208/312 retained.
Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 block256,no model/
context/clock override,packing/reference excluded,two counterbalanced nine-
sample process repeats.208 K32768 branch-clean135.141/135.245 ->137.904/
137.986TF,clock-adjusted88.52/88.55% ->90.65/90.63%;K8192 130.357/130.455
->132.637/132.536.104 long122.401/122.444 ->127.453/127.430;312 long139.941/
139.782 ->141.588/141.588;large1664CTA145.773/145.897 ->146.550/146.436
(~97.1%). Net improvement supports scheduling intuition,not proof of zero
publication-wait stalls/fixed192-cycle overlap. Freshmfma-first namespace
preservesbranch-clean before code/data. Small entry point reuses existing
matched driver; historical branch driver/audit pin completed snapshots.
Source/native/raw data in `native-gemm/results/mfma-first/`,order proof in
`results/mfma-first-audit/`,notes MFMA_FIRST.md. No packet interleaving or
further scheduling change,production/CMake/model/system/clock-policy change.

**FP8 B balanced all-wave producer (user-directed):** FP16 A/finite E4M3FN
B,convert once before unchangedFP16 LDS/consumer. Repack BOTH inputs so wavew
produces K8 slicew of all128 rows/columns: two contiguousA x4+oneB x4 load
per wave/stage, no role branches. A8KiB+B4KiB/CTA/stage vs8+8KiB; separate
allocations. In-place B expansion backwards,high output first: four source
dwords->eightFP16 dwords,32VALU/wave/stage, exact earlier perm/PKASHR/AND/
PKMUL256 sequence,IEEE subnormal mode. Same16fixed producerVGPRs; two source
pointer pairs usev40:43. Native147-instruction two-stage backedge vs81,
32MFMA/6global/8write/16read/16eachconversion opcode, unchangedfences/control.
140VGPR/64AGPR/46SGPR vs136/64/46,16KiB LDS/private0 and cap3CTAs/CU.
First schedule is a conversion burst after2Astores and before2Bstores/future
prefetch/remaining3q0MFMA. No further rescheduling attempted.
Checks:524288 padded-coordinate pack elements,4064 mixed in-place GPU input
vectors/65024 half outputs (all254finite codes everybyte position,signedzero/
subnormals),18 full-GEMM finite-code tests(K1..64,A+/-1/16 exactint64oracle),
312 matched checked launches(30before+30afterfunctional+252timing),88hashed
processes all pass. Generated29-stage-count compute/raw-read/fence projection
matches, native math/read order matches(addressinputs normalized). New shared
host bridge's normal128/256 GPU.text byte-identical to preservedMFMA-first.
Measured nativeHIP O3 gfx90a HIP0/BDF31:00.0 CPU+memoryNUMA0 block256,no model/
context/clockoverride,packing/reference excluded,two counterbalanced9sample
repeats. NEGATIVE lowgrid result:K32768 104FP16 127.407/127.476 ->FP8 120.750/
120.562TF;208 137.972/137.918 ->134.038/134.037(~88.10%);312 141.693/141.416
->139.021/138.966(~92.14%);large1664CTA146.486/146.513 ->146.473/146.455
(~97.4% clockadjusted). Consistent with longerproducerpath,not isolated
conversioncost/unique bottleneck proof (packing,addresses,registers alsochange).
Separatefp8-b variant,notpromoted;normal defaults nowfreshfp16-bridge namespace
to preserve measuredmfma-first sources/binaries. Actualdriver/source/native
and gate records in `native-gemm/results/fp8-b/`,projection in `fp8-b-audit/`,
notesFP8_B.md. Artifacthandling: objcopysectiondump withoutoutput rewrote two
.hsaco containers; measuredexecutables untouched. Originalcodeobjects re-extracted
from thoseexecutables,matched alreadyarchivedSHA256,andrestored,notrebuilt or
relabeled. Final allhistorical/native hashes pass. No production/CMake/model/
session/system/package/clock changes. Unrelated dirty/userrelocation unstaged.

**FP8 copies/early B load + MFMA interleave (user-directed, two measured steps):**
Eight permutations copy v60:63 to v52:59, then immediately issue next B x4
load into the freed raw buffer. Add four fixed VGPRs to avoid overwriting
current conversion data; same packing/geometry/fences, A prefetch stays late.
Then move current q0/C01 after six arithmetic VALUs, before the remaining18.
C00 remains immediately after VM0; C10/C11 remain after stores/A prefetch.
Write-to-publication window shrinks; net gains survive, not zero-stall proof.
Both new147-instruction loops retain32MFMA/6global/8writes/16reads/64convert/
6waits/4barriers/3control; early->interleaved is EXACTLY two native C01 moves,
every other native instruction/order unchanged. Both144VGPR/64AGPR/46SGPR,
private0/16KiB LDS/runtime cap3 unchanged; first step140->144VGPR is explicit.
Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memoryNUMA0 block256,automaticclocks,
no model/context/override,packing/reference excluded,two counterbalanced9sample
repeats per A/B. K32768 step1 burst->early TF:104 120.687/120.896 ->119.776/
119.940 (mixed,-0.77%);208 134.063/134.179 ->134.515/134.658;312 139.067/
139.141 ->139.132/139.160;large146.494/146.482 ->146.817/146.821. Step2early
control->interleave:104 119.796/119.755 ->124.586/124.631 (+4.04%);208134.619/
134.554 ->136.676/136.796 (+1.60%,~89.98%eff);312139.141/139.123 ->140.978/
140.864 (+1.29%);large146.788/146.785 ->147.333/147.364 (~98.0%eff). Two
adjacent matched A/Bs,not a randomized3way or fresh FP16 comparison.104/208/
312 pluslarge retained atK8192/32768.624 checkedlaunches +36fullfinitecode
GEMMchecks +2 GPU source-reuse gates(65024halfvalues each). Gate poisons raw
buffer immediately aftercopies, beforearithmetic.29stagecounts match compute/
read/fence projection,3Nglobal/4Nstores/8Ncopies,nooverfetch; stages-1C01moves.
Explicit regeneratedburst asm matches archivedoriginal. Separatefresh
fp8-schedule/{early,interleaved},dataresults/fp8-{early,interleaved},order
proof fp8-schedule-audit,notes FP8_SCHEDULE.md. Historicalaudit/driver pin
completed snapshots. No further tuning or production/CMake/model/system/
package/clock-policy changes; no oldrawdata or executable overwritten.

**K128/N128 FP32 scales and192-register triple accumulation (user-directed):**
SUCCESS at256 architectedVGPR/0AGPR/51SGPR/private0/16KiB LDS, actual2CTA/CU.
Kept all five raw operand slots and20 producer registers. SGPR global bases
plus shared lane offset and compact LDS address state replace vector pointer
pairs/external vector inputs; output/stamp stores inside asm remove C++ vector
lifetimes across it. Workv0:63,scaledC64:127,rawA128:191,rawB192:255. No spills,
AGPR copies or reduced pipeline depth. First K8 uses src2=0 per rawmicrotile;
previous rawbank scales via64 direct FP32FMAs during newgroupq1 MFMA, final
partial scales after conservative64cycle tailgap. FP32 table[K128][N128],
oneSGPRscale/wave/group,reused acrossM; partial/zeroK guarded, nooverfetch.
Independent int64 logical partial +orderedstd::fma oracle; ones/dyadic/non-
power-of-two/zero metadata, finiteBpattern andlong/tailK all pass.683 checked
launches/235hashed processes (504timings+179correctness),rawclocks/finiteC/
poison/guards;29stagecounts track bankownership/scaleIDs/contributions/order.
Controls: compactunscaled128regs cap4 vs byte-identicalreserved256 cap2,
thenactualscaled256. Matched nativeHIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory
NUMA0 block256 automaticclocks no model/context/override,packing/reference/
scaleupload excluded,two9sample repeats/reversedvariant+problemorder.
K32768 original->compact TF104124.498/124.542 ->125.302/125.325,208136.689/
136.689 ->137.375/137.267,312140.911/140.921 ->141.416/141.406,large147.383/
147.401 ->148.116/148.130: no compaction performance loss. Reserved256:
125.055/125.212,137.186/137.240,139.689/139.689,144.905/144.910. Actualscaled:
118.757/118.939,135.851/135.824,137.950/138.050,142.521/142.512. Extra scaling
vsreserved costs~5.02/1.00/1.21/1.65%; not uniquelyattributed toFMA latency
(metadataSMEM atgrouphead,bankreset/drain/finaltail alsochange). BothK8192/
32768 preserve104/208/312pluslarge. Source/native/data inresults/scaled,
proofscaled-audit,notesSCALED_FP8.md. Oldunscaled FP16tiles/FP8 GPU.text remain
byte-identical through sharedhostbridge; fresh defaultbridge namespaces,
historicaldrivers pin originals. No production/CMake/model/session/system/
package/clock-policy changes or measuredbinary overwrite; noobjcopy.

**PackedFP32 moves/FMA (user-directed):** gfx90a v_pk_mov_b32 initializes two
consecutive VGPRs and transfers timestamp SGPR pairs;128zeroMOV->64packedMOV,
8timestampMOV->4packedMOV. Scale microtile16scalarFMA->8packedFMA,hot8stage
128scalar->64packedFMA,final64->32. No scheduling/tile/packing/fence change.
Aligned s14:15 is legal; previous s15 selectedhigh BOTHhalves(op_sel010,
hi111),current s14 selectedlowBOTH(000,hi101). Odd s15:16 is rejected for
alignment; noVGPRscale copy. Allvariants256VGPR/0AGPR/51SGPR/private0/16KiB
andactualcap2 unchanged. Three-way matchedscalar/mov-only/mov+FMA. NativeHIP
O3 gfx90a HIP0/BDF31:00.0 CPU+memoryNUMA0 block256 automaticclocks,no model/
context/override,packing/H2D/referenceexcluded,two9samplecounterbalanced
repeats. K32768scalar->both TF:104118.817/119.040 ->120.002/120.147(+0.96%);
208135.798/135.798 ->136.944/136.809(+0.79%);312137.941/138.041 ->139.114/
139.086(+0.80%);large142.594/142.554 ->143.490/143.517(+0.65%). Movonly
withinnoise.104/208/312 retained alsoK8192.568checkedlaunches/232hashed
processes,nonpower/zero/nonuniformscales,longK/tails/seeds/finiteB allpass.
Alloutputs poisonfinite/guards,actual64bittimestampdumps validated. Exact
fullgenerated/nativeflow equalsbaseline afterpackedexpansion andbranch-target
normalization;alignedpairs/samescaleperhalf checked.29stagebank/scalehistory
pervariantmatches. Freshscaled-packed namespace,oldsources/binariespreserved;
historicalrun_scaled pinscompleted sources. Dataresults/scaled-packed,
auditresults/scaled-packed-audit,notesPACKED_FP32.md. No2xkernelthroughput
claim or isolatedinstructionlatency proof;mostotherworkunchanged/overlap
alreadyexisted. No furtherinterleaving/scaleprefetch or production/CMake/
model/session/system/package/clockpolicy change. Researchlog remainsunstaged.

**CORRECTION: packedtimestamp selectors and complete rerun.** Final binary
classification check exposed repeatedlow/high words. V_PK_MOV defaults select
low32 for BOTHdestinations,unlike V_PK_FMA packedhigh defaults. Priorpacked
trial clockmetrics AND its expansionproof are INVALID (auditor modeledwrong
default,ratio-onlyclockcheck was circular). OriginalGEMMrefs passed andHIPevent
is separate,butentiretrial superseded;raw/source/exes/proofs retained with
STATUS.json/INVALID_CLOCKS.md. Explicit op_sel:[0,1] fixes timestampcopies.
DedicatedGPUrawbit gate74048copies passes,defaultselectornegativecontrol
rejects73920 outputs. New independentbodyspan<=HIPevent guard rejectsall28
oldaffectedprocesses/252samples and accepts14scalarcontrols. Correctedexpansion
audit models V_PK_MOV separately; source/native matchscalar afterexpansion.
Freshscaled-packed-fixed binaries/data/full568checkedlaunches/232processes
allpass stricterrawclockguard,nonpow/nonuniform/zero scales/tails/seeds/longK/
finiteB. K32768scalar->both TF104118.757/118.898 ->120.271/120.292;
208135.824/135.758 ->136.944/136.944;312137.941/137.868 ->139.104/139.012;
large142.481/142.548 ->143.433/143.431.256VGPR/0AGPR/51SGPR/private0/cap2
unchanged. PackedFMASGPRselection was correct;bugonlytimestampmove defaults.
Oldclockdumpbytes neveredited;git*.bin binaryattribute preventscontentbased
textmisclassification,not datacorrection. Correctedbitgate/proofs/logs in
results/scaled-packed-fixed,genericverifier warnsoldclocktrialinvalidwhile
checkingitsimmutablehashes. No production/systemchanges or originalmeasured
exeoverwrite. Thisentry supersedes previouspackedclock-validation claims.

**Three scale-latency proposals implemented, measured, stopped (user-directed):**
Late current-scale load issues after previous-bank q1 drain,not beforeheadLGKM0.
Existing next-stage LGKM0 establishes readiness;finalexplicitwait covers one-
stage partial groups. NoadditionalVGPR/nohotfence or payloadchange;256/0/51,
private0/16KiB/cap2. Batch repacksmetadataN-major,K-padded-even andloads one
SMEMpair per2groups intoSGPR20:21 (53SGPR),readiness/low-highcopies explicit,
paddedwordneverapplied. Read variant is late+q1nextoperand reads beforelast
packeddrain;not basedonbatch. Keeps SrcC/destination bank handoff unchanged.
Four-way matched HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memoryNUMA0 block256,
autoclocks/no model/context/override,packing/H2D/reference excluded,two9sample
reversedvariant+problem repeats. K32768 baseline->late TF104120.331/120.619
->125.390/125.454(+4.11%);208136.927/136.978->137.277/137.328(+0.26%);
312139.194/139.098->139.565/139.590(+0.31%);large143.428/143.443->144.631/
144.634(+0.83%).104K8192+3.23%. BatchvsLate mixed (208long-0.12%,104short
+0.30%,large+0.05%);ReadvsLate mixed (104short-0.23%,large-0.04%,312tiny+).
No consistentextra win;notpromoted, no further sweep. All3implemented,789
checkedlaunches/341hashedprocesses (504timing+285functional),104/208/312plus
large retained atK8192/32768. Nonpow/zero/nonuniformscales,longK/tails/seeds/
finiteB refs/poison/guards/independentclockspan/rawdump checks pass.29stage
counts explicitly commit pendingSMEM onlyatLGKM0,trackorderedscale/rawbank
IDs,pairpadding,3Nweights/4Nstores/8Nreads;everyrequesthas8MFMAleadbeforewait.
NativeLatehot loop equal exceptmetadata retiming;wholeNativeRead differsonly
q1read-pairmovements. Hostbridge FP16(128/256),unscaledFP8 andbaselineScaled
wholeGPU.text byte-identical. Newscale-latency namespace/archives,oldbinaries
untouched. Source/native/dataresults/scale-latency,proofscale-latency-audit,
notesSCALE_LATENCY.md. No production/CMake/model/session/system/package/
clock-policychanges or furtheroptimizations. Researchlog/userrelocation unstaged.

**User-directed result-layout cleanup:** removed all12 sibling audit/audit-final
folders under native-gemm/results. Moved their files into owningexperiment/
checks; latest JSON ->summary.json,distinctinitial ->initial-summary.json.
All89 original filecontents verified bySHA256 after moves,3 identical header
copies deduplicated. StageK64 files nowbranch-clean/checks/gpu64 andmfma-first/
checks/gpu64; activeverifier checks relocatedmanifests(require2 each),allnative/
source/executable hashespass. Audit scripts nowemitchecks/summary.json;
intermediatereruns usegitignoredtemp. Docsupdated,immutableas-ran source/driver
snapshotsretainoriginalpathstrings; no legacysymlinks. CPUlatestlatency/packed
checks rerunintemp pass. InvalidpackedtrialremainslabelledINVALID initsown
experiment. No measuredclock/CSV/sourcebytes or binarieschanged; noGPUcode/
production/systemchanges. Unrelateduserrelocation/dirtyworkleftunstaged.

**User-proposed earlyA/balancedq0 experiment:** movedfutureA loads out oflate
conversiontail;balancedlayout implementsactualnativeq0 gaps13/15/15 vsold
19/24/0(earlyAonly21/22/0),same43interveninginstructions,notall4cycleVALU.
Userreasoning identifiesunusedissueoverlap;no exact60cycle/OOOclaim. Aregisters
v44:51 freedby2LDSstores,Bconversionv52:59 independent;allBperms beforefuture
Bloadv60:63. Balancedconversion15/9split andpointeradds afterBstores,adjacent
64bitcarrypairs. No extraops/fences/regs/packing/bankchange;256VGPR/0AGPR/
51SGPR/private0/16KiB/runtimecap2. Futurevalidity guards/tails unchanged.
Two9samplecounterbalanced3way,HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memoryNUMA0
block256 autoclocks/no model/context/override,packing/H2D/refexcluded.
K32768latebaseline->balanced TF104125.618/125.528->127.920/127.874(+1.85%);
208137.348/137.321->139.003/139.072(+1.24%);312139.708/139.773->141.397/
141.406(+1.19%);large144.745/144.749->146.174/146.176(+0.99%). EarlyAalone
flat104,negative208/312/large~-0.30/-0.36/-0.50%. K8192balancedalsoimproves,
104+1.37%. No independentearlyloadwinclaim,no additionaltuning.568checked
launches/232hashedprocesses pass scale/ref/finiteB/longK/tails/seeds/poison/
guards/rawclock+independentHIPeventcontainment.29stagecounts checkpending
stagingpacketlifetimes,copy/reloadownership,exactA/Bpointerstreams,conversion/
store/matrix/read/fence/scaleorder andinstructionmultisets. NativeearlyAonly
restoresEXACTbaseline bymovingtwoloadsback;balancednativeprojections/counts/
resourcesunchanged,13/15/15asserted. Recordsresults/q0-schedule withonechecks/
summary.json;intermediatererunstemp,no siblings. Oldscale-latencydriver pins
originalsnapshots;no measuredsource/data/binaryoverwrite. NotesQ0_SCHEDULE.md.
No production/CMake/model/session/system/package/clock-policychanges.

**Backlog review/commit (user-directed):** commit the research-log relocation
and accumulated status history; all884 nonblank lines removed from the design
document remain verbatim and ordered here. Retained source/binary hashes for
older CU-load and unloaded/window-latency studies pass;1134 latency/1440window/
240counter summaries and provenance reproduce byte-for-byte in fresh temp.
CU ISA check49 kernels/325 premature-read negative controls and a64-sample
short GPU smoke pass. LegacyMFMA chain/loop-shape sources compile and pass
all built-in exact per-thread gates in fresh builds,private0; their comments
remain historical hypotheses,not a new hardware model. Baseline latency runner
now refuses nonempty outputs and requires matching build hashes,as the window
runner already did; rejection tests pass,old measurements unchanged. Local
external tool checkout/venv remains ignored,not vendored. User deleted the
handwritten assembly sketch;it is not restored or committed. No production/
model/CMake/system/package/clock-policy changes from this review.
