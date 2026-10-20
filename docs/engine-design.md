# Alice Inference Engine — Design

Status: living document. GPU-resident execution is the only path; CPU
bring-up is retired. AGENTS.md tracks the machine and state of play.
Engine owns generation, host sampling, tokenizer metadata, and one worker /
Model per tensor-parallel rank. Factories shard weights; TensorView supports
strided host shards. Model state and the FP32 residual stream stay on device.

**TP collective integration (2026-10, user-directed, §17):** every block
publishes an immutable slot in a rank-local contribution slab and reduces
out-of-place into trunk io. S=1 uses native device-flag P2P: fused single-CTA
for <=16 KiB rows, pointer-hoisted/dephased data-major 2×1024 gather plus
ready barrier for the 32 KiB 70B row. S>1 uses out-of-place RCCL prefill.
There is no consumed epoch or copy-back. All workers complete GPU execution
before the next forward reuses slots. P2P epochs remain monotonic across
RCCL prefill gaps, graph replay, and reset. No rocSHMEM dependency.

Canonical's RCCL build is still broken. CMake selects the existing validated
AMD 7.1 overlay through `ALICE_RCCL_LIBRARY`; executable build RUNPATH and a
runtime symbol-path guard prevent accidental use of the wrong build. Engine
creates one load-phase communicator ID; workers initialize their own rank's
communicator after all models load, then record the S=1 compute graph.
`WorkItem` contains only execution commands/tokens, never initialization
payloads. Prefill remains eager; graph replay and rank threads are retained.

Named buffers are fixed-size inter-layer communication state, not loading
workspace. Their table is their sole owner; no grow API or duplicate ledger
registration. ROCm weight constructors own scoped device repack staging and
release it after completion during loading. All runtime work names the rank
execution stream; temporary upload/repack reuse is explicitly ordered.

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
and runner cases. Direct-to-LDS is the sole producer; only one/two LDS buffers
remain selectable. Five GPU entries/build (two GEMM, two LDS diagnostics,
one copy validator) replace eight. Six builds/30 entries, 340 exact checked
launches and two removed-selector rejection checks pass; all 24 surviving
benchmark core instruction/register lists match after branch/address
normalization. Cycles per four-wave MFMA step are now the primary report:
K32768 executes 4096 steps/wave, so metric=CTA core span/4096, including initial
staging, compute/waits and stage synchronization, not isolated MFMA latency.
`body_us` measures that region with the steady counter; `launch_us` is the
full event interval including epilogue, neither includes host packing/upload
or verification. Fresh StageK128/direct/two-buffer runs are 143.252/143.403
cycles/step (~347 us); no new speedup or physical-unit claim. Historical reg
results remain archived; no production/CMake/model changes.

## 1. Purpose & target

A from-scratch C++20 LLM inference engine for this machine: 4× MI250X
(8 GCDs × 64 GB HBM2e, gfx90a, ROCm 7.1) behind a 48-core Zen3 host.
Relatively high-performance and relatively general: built for this
hardware first, structured so other GPU vendors can be added later
without rearchitecting.

Code quality is a feature, not a preference. Development is agent-heavy,
and ill-shaped code compounds under agent edits into "playdough": layers
of patches and one-time constraints squeezing the project out of shape.
The codebase must stay small, readable, and self-documenting; the tech
stack (CMake, C++, vendor libraries, hipcc) is chosen by what produces
good code and good results, never the reverse, and no decision is kept
past the moment it is demonstrably wrong.

Bring-up models: Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-Distill-Llama-70B (HuggingFace sources under `/opt/models`,
packed into `/opt/models/alice/<name>/` by `tools/pack_model.py`). The 70B
(141 GB bf16, 80 layers, 17 shards) is the sizing reference.

## 2. Deployment assumptions

Single user. One live context at a time; every new input is appended to
*the* context. No prefix matching, no rollback requests at the engine
level. Consequences: no multi-context registry or locking; stateful
Blocks own their state directly, sized once for max context; state ops
reduce to write-during-forward and `reset_state()`; memory planning is
fully static at load. If multi-session ever arrives it is bolted on by
instantiating state per session — compute never rearchitects, because
state lives behind Block boundaries.

## 3. Execution model

The engine requires a GPU. There are no CPU-only builds.

All model state lives on the device: the residual stream, the block io
buffer, attention KV, norm weights, RoPE tables. All state operations —
normalization, residual adds, position encoding — execute on the device.
A forward pass performs no host↔device transfers inside the block loop;
the regular traffic is token ids (in at the embedder, out at the sampler
— four bytes per step). The host loads weights, tokenizes, detokenizes,
and runs the frontend.

The stream is FP32 end to end. The numerics boundary that was paid for
during bring-up is kept — only GEMM operands are quantized (bf16
storage, fp32 accumulate). What changed is the stream's address space,
not its precision.

Blocks are written for their target architecture, in that architecture's
native language and libraries. There is no vendor-compatibility shim
inside Blocks, because optimized-kernel reality does not abstract:
FlashAttention has a different best implementation per SM generation,
wgmma source does not compile across adjacent NVIDIA architectures,
NVFP4 is native on sm_100 and emulated elsewhere, and the right tensor-
parallel strategy depends on link topology. A shim would leak at exactly
the points that matter. Portability is instead achieved by writing
several native implementations of one abstract block class and selecting
at load the one matching the device present. Vendor BLAS is used where
it genuinely covers the problem (dense GEMM) and not where it doesn't
(attention, fused elementwise chains, quantized formats). Measured
addition (2026-09, research/decode-roofline): S=1 GEMV is a "doesn't"
— rocBLAS gemm_ex picked no-split-K kernels running the FFN shapes at
313-727 GB/s against a 1290 GB/s read ceiling (26 of 29.6 ms/token).
The gpu_rocm family now carries its own split-K GEMV, dispatched at
S==1 inside both blocks; see §15.

The Trunk's own operations are the deliberate exception: norms, residual
adds, and RoPE are plain SIMT kernels — no matrix intrinsics, no warp
tricks — written once against a small macro header that maps kernel
spelling onto HIP/CUDA/MUSA-flavored compilers. That header exists for
this level only and never crosses a Block boundary.

CPU Blocks survive in exactly one role: development reference. A CPU
reference block implements the same Block interface by copying its input
to the host, computing with naive, maximally readable code, and copying
the result back. Its purpose is fault localization in research-grade
kernels: a correct Flash Linear Attention implementation is a research
project of its own, and comparing a full GPU model against a full CPU
model cannot isolate which block diverges — swapping one block instance
at a time can. Transfer cost is irrelevant because such blocks never run
in production. One practical exception may eventually appear — CPU-
resident MoE experts — and will be designed for if it does.

Device code is compiled with the vendor toolchain (hipcc on this
machine). Runtime-compiled string kernels (hipRTC) are retired: they
existed to keep hipcc out of the build, and the build no longer avoids
hipcc.

## 4. Model = program, .model descriptor = contract

The engine loads via the alice model format: a packed `.model` file
(one JSON object + `\n\0`) + a sibling `weights.bin`. The JSON's
`model` map is the **descriptor** — composition (block specs, members,
blocks), each carrying only a content-ID (the BLAKE3 of its bytes); the
unsigned `files` array maps IDs to `{file, offset, size, codec}`
locations. The descriptor is the *total contract*.

Identity is separated from location: a block/member record carries only
its content-ID; the ID doubles as the integrity check (the DataLoader
verifies `BLAKE3(bytes) == id` on every fetch, so no caller ever sees
unverified bytes). Content-addressing dedups — identical bytes share one
ID and one stored copy (tied weights: the lm_head names the embedder's
ID, the one legal shared claim, fetched twice but stored once). The
factory's `derived_bytes()` is the two-watches check: it derives the
expected record size from the spec and reconciles it against the fetched
blob in `construct()`.

Sharding is physical packaging, invisible to the contract: the packer
(`tools/pack_model.py`) consolidates HuggingFace shards into one
`weights.bin`; no HF `index.json` is read at load time. The packer is the
only producer of `.model` files.

## 5. Tensor & dtype contract

The typed tensor `Tensor<T>` is the **host-side vocabulary** (2026-09
revision: the former non-owning `TView<T>` and the loader's `Staging`
merged into one move-only owning type — exactly one owner per buffer,
one live handle, "freed on return" is mechanical rather than a caller
convention). Owning, rank ≤ 2, row-major, PACKED — deliberately not
strided: transposes are explicit decisions at GEMM call sites (rocBLAS
op flags), never metadata. The strong `bf16` struct. Runtime `DType`
survives only in file-facing structs; the descriptor's block specs are
dtype-free, the expected dtype being derived from the block type and
enforced at the factory's `construct()` (the load-boundary check is the
DataLoader's BLAKE3 verify, not a dtype check). Quantized formats get
concrete composite types (data + scales), never a polymorphic hierarchy.

The device-side stream and device buffers are not Tensors. They are raw
device allocations owned by components, referenced as plain pointers in
device code; borrowed host access at compute boundaries (oracle kernels,
the host Sampler) is likewise ptr + dims, mirroring the `gpu::Block` io
contract.

## 6. Component taxonomy

- **Functional / Weightful / Stateful / Block** — as before: stateless
  primitives; weight owners; state owners (KV, SSM state); the Block is
  the smallest unit the descriptor describes.
- **Embedder** — token ids → stream rows (a device-side gather).
- **LmHead** — final hidden → logits (device GEMM; last row only at
  prefill). Deliberately not bent into the Block signature.
- **Sampler** — logits → token id, on device; owns RNG state. The
  sampled id is the only per-step D2H unless a frontend asks for logits.
- **Trunk** — owns the device-resident stream buffers, ALL norm weights,
  the norm/residual SIMT kernels, and the wiring scheme. A model with a
  new residual scheme gets a new Trunk, by design.
- **Model** — container + wiring loop; owns parts via `unique_ptr`;
  knows no architecture specifics. Token ids in → token id out.

## 7. Forward loop

```
io = embedder.embed(token_ids)          // gather into the fp32 stream
for block in blocks:
    contribution = slot(i) if TP > 1 else io
    trunk.read(i, contribution)          // residual → contribution, apply norm
    block.forward(contribution)         // in place; publish immutable TP slot
    if TP > 1: allreduce(slot(i), io)    // produced barrier, gather → local io
    trunk.write(i)                      // residual += io
token = sampler.sample(lm_head.project(trunk.output()))   // on device
```

Prefill processes the entire prompt including the last token. Invariant:
every token enters Block state exactly once, and its logits are
available immediately after processing. Position never crosses any
interface; each stateful Block keeps a private position counter and may
evict/compact its state at will — strictly its own business.

## 8. Class structure

### 8.1 Blocks

- `Block`: `forward` over the shared device stream, in place; virtual
  destructor. `StatefulBlock` adds `reset_state()` plus default-trap
  snapshot methods (future session API).
- One abstract class per (block type × weight dtype), unchanged: dtype
  dispatch stays at the type level, so no implementation carries
  per-dtype switches with per-architecture ifdefs.
- Concrete implementations are `final`, one per target architecture
  (`DenseFfnBF16Gfx90a`, …). A factory selects the implementation for
  the device present. **There is no fallback chain terminating in a CPU
  impl**: if nothing matches the present hardware, loading fails loudly.
  Silent downgrade was a bring-up convenience; in production it hides
  configuration errors.
- CPU reference blocks (retired): one naive host implementation per
  abstract block class, swappable in for development (§3), was the bring-
  up oracle. Self-contained — the D2H/H2D copies lived inside the
  reference block, so the Block interface stayed uniform and nothing
  else knew they existed. Retired 2026-09: naive reference impls for new
  block types are now done as GPU kernels, not host blocks; the host-side
  math helpers survive as `src/host_math`. The Block interface stays
  uniform; nothing else knows the impl exists.
- Self-containment, unchanged: private weight storage, private math, no
  shared kernel toolbox across impls. Constructors take already
  validated staged weights (`Tensor<T>`, moved in), copy/convert/pack
  into private layout (staging H2D included); the staging dies inside
  the constructor, so "freed on return" is mechanical. Contract
  amendment (2026-09): few-role blocks take single staged tensors
  positionally; the trunk — the one component whose weights are a
  homogeneous set — takes the staging vector itself.

### 8.2 Trunk

Role unchanged (the "where the mess lives" class), now device-resident:
owns the stream buffers, norm weights, and the plain-SIMT norm/residual
kernels of §3. First implementation: RMSNorm + add-residual. Revisit
triggers for merging norm into Trunk: trunk-impl proliferation,
mixed-norm models.

### 8.3 Model, Embedder, LmHead, Sampler

Thin concrete classes; no dtype×arch abstract matrix until a second
implementation exists. Sampler config fixed at construction (temperature
0 = argmax for tests); the Sampler owns the RNG — acceptable mutable
state under single-session.

Model's shape (2026-09): named members over forward-declared incomplete
types — the "exploded pimpl" (one Impl indirection was tried and
rejected; `private:` leaking layout is solved by not including the
component headers, not by hiding them behind a pointer). The public
header stays free of internal- and vendor-includes (no trunk.hpp, no
HIP/rocBLAS); their headers are included only by the implementation
TU. Consequence, accepted deliberately: a change to the component SET
edits the public header; a change inside a component never does. The
GPU Context (device binding + rocBLAS handle) is owned by the Model,
like the shared scratch — session-scoped GPU resources live under
Model, never in process-wide state.

### 8.4 ModelLoader

Narrow entry that sniffs format by content and delegates
(`SafetensorsLoader` now); registry machinery deferred until format #3.

## 9. Memory architecture & weight loading

- No allocation during `forward`; everything is sized at load.
- Private buffers per component — device allocations now: weights,
  state, transients.
- Named buffers allow layers to communicate beyond the residual stream.
  Same name + size returns the same stable pointer; any size disagreement
  fails loudly. They never grow and live until model teardown. They are
  separate from the block arena and attention scratch, and are not a
  temporary-workspace allocator.
- Column-major weight repacking uses constructor-local device staging
  (`hipMalloc`, scoped `hipFree`), released after repack completion during
  loading, including on exception. No repack staging survives into inference.
- The shared stream buffers (residual, io) are device allocations owned
  by the Trunk, `[max_prefill_chunk, d]` fp32. Under TP, Model additionally
  owns a contribution slab `[n_blocks, max_prefill_chunk, d]` fp32 per rank
  (no slab at TP=1). Each block normalizes into and computes in its slot;
  once published, the slot is immutable through forward completion.
  Allreduce reads peer slab bases plus the same fixed slot offset and writes
  local trunk io, never a published slot. Slab allocation checks current
  free memory and reports its footprint / max_prefill_chunk on failure.
  One more cross-block sharing instance exists (2026-09, first §14 scratch-pool use): the v4
  prefill attention partials buffer, loader-allocated once
  (`nh × max_prefill_chunk × ceil(max_seq/64) × (hd+2)` fp32) and reused
  by every attention block — each block's partials are produced and
  consumed within its own `forward` (split → combine).
- Load flow, per block in order: `pread` shard bytes → pinned host
  staging → H2D → staging freed. Peak host RAM ≈ one block's staging.
  Pinned staging is mandatory (pageable H2D runs a driver-staged path,
  measured 2.2 vs 28.5 GB/s); the pinning API lives inside the GPU
  platform module, not in core code.
- **No mmap anywhere.** Weight streaming is one-time, not re-read; the
  page cache would duplicate data and add RSS/reclaim pathologies. If
  the SSD-resident regime ever arrives, the answer is an explicit
  user-space cache (the Colibri playbook), designed then.
- Multi-die: the 70B needs ≥3 GCDs. Blocks get a placement; the stream
  crosses between devices by peer-to-peer copy at placement boundaries.
  A named seam, built when 70B tuning demands it.

## 10. Numerics, determinism, threading

- Numerics policy (user-signed): cross-implementation equivalence is
  **tolerance-based**; within-impl determinism holds per (impl, S,
  kernel schedule). The teacher-forced canonical test vs HF is the
  equivalence gate.
- Device-side determinism: fixed kernel schedules; vendor BLAS invoked
  with fixed algorithms (no run-to-run autotuning).
- The model loop is strictly sequential; intra-op parallelism is each
  implementation's private business. A forward work item completes GPU
  execution before its worker reports done: logits paths use async D2H plus
  synchronization on the rank stream, and `ForwardRun` synchronizes that
  same stream. The Engine's
  all-worker completion barrier therefore protects contribution-slot reuse
  across forwards. No host synchronization is added inside the block loop.

## 11. Extensibility targets (designed for, not built)

- Quantized weight formats (MXFP4, NVFP4, FP8-B128): a format is a
  bundle of tensors plus compute semantics, landing as new values on the
  block-type × dtype axis — with per-arch native implementations (NVFP4
  native on sm_100, emulated elsewhere; exactly why Blocks carry no
  shim). The constructor contract already permits dequantize-at-load
  fallbacks; the two-way audit gains format-specific shape relations.
- New block types: MoE, linear/hybrid attention (KDA/FLA — expect the
  CPU-reference-block workflow of §3 to be the bring-up path), SSM.
- New trunk schemes: mHC (GLM), attention residuals (Kimi).
- Multi-die placement (§9). CUDA/HIP graph capture when decode
  approaches the kernel-launch-latency floor (~150–200 launches/token).

## 12. Non-goals

- CPU-only operation (abandoned 2026-09).
- A vendor-abstraction layer inside Blocks (§3).
- Tokenizer / chat template / generation loop inside `Model` (frontend).
- GGUF. Multi-request scheduling. Fork/beam/speculative state ops.
  Python in the engine core.

## 13. Testing strategy

- The equivalence gate: teacher-forced canonical logits vs HF per model
  (`tests/canonical*/`, gate max|Δlogit| < 0.75). Watch item: max drift
  grows with depth (0.22 / 0.28 / 0.68 at 16 / 32 / 80 layers).
- **Reference quality (user-signed 2026-09)**: short fixtures stay bf16-
  reference with the 0.75 max gate. Long fixtures (~1500-token WikiText
  prompts, regenerable via `dump_canonical.py --prompt-file`, gitignored)
  use an **fp32-ideal reference** (torch `dtype=float32` upcasts weights
  exactly and runs the stock graph in fp32 — the mathematical ideal of
  our fp32-stream design; measured: the CPU engine is EXACT against it,
  max 0.0002, while PyTorch's own bf16 inference sits 15.6/1.8 max away).
  Gates on the long case: GPU suite mean ≤ 0.05 AND max ≤ 0.75... (the
  bf16 GEMM-input budget at massive-activation positions; observed 0.018
  / 0.76 / 1.52). (A temporary host-path CPU long suite gated at 0.01
  proved the semantics exact — 0.0002 — and was retired with the host
  path; the proof stands recorded here.)
  Greedy decode comparison is exact-match with a coin-flip rule: a
  divergence is a bug only if our token is ≥0.25 below the reference
  argmax (near-ties flip legitimately within the bf16-input budget).
- **Single-block HF oracle (2026-09)**: `tests/python/dump_hf_block_oracle.py`
  extracts real weights from a chosen layer and runs the exact HF
  `LlamaAttention` / `LlamaMLP` submodules on synthetic data.  The C++
  test constructs the corresponding alice::gpu_rocm block with those
  weights and asserts against the HF output.  Because only one block is
  exercised (no layer-stack amplification), the bound is much tighter
  than the full-model gate: observed max|Δ| 0.040 prefill / 0.003 decode
  for attention, 0.016 prefill / 0.010 decode for FFN (Llama-3.2-1B
  layer 5, S=8/S=1).  This is the primary per-block numerical oracle.
- **CPU reference (§3)**: retired. The `cpu_generic` host-block reference
  impls are gone (naive reference impls for new block types are now done
  as GPU kernels, not host blocks); the `ALICE_CPU_REF_BLOCKS` dev
  fault-localization swap-in retired with them. The host-side math
  helpers (RoPE tables, rmsnorm/add_rows/silu — the kernel-level oracle
  for `test_gpu_kernels.cpp`) survive as `src/host_math.{hpp,cpp}`.
  `test_gpu_blocks.cpp` keeps only structural tests (incremental==
  batched, reset freshness) that have no HF equivalent.
- Determinism: same tokens + fresh state + temperature 0 → identical
  output.
- Loader: the DataLoader's BLAKE3-verify path (corruption = hash
  mismatch, fails loud); the real packed `.model` files as fixtures.
  Ad-hoc models validate with zero recompile via `CanonicalEnv`
  (`ALICE_CANONICAL_{MODEL_FILE,FIXTURES[,CHUNK]}`).
- Slab allreduce: four-rank kernel test covers separate output, unchanged
  published slots/padding, tiny fused / 32 KiB gather paths and tails, and successive
  forward epochs. TP=2 canonical validation covers wide prefill, S≤4 and S=1,
  reinitialization, and chunked `generate()` with `ForwardRun` completion.
- Tests require a GPU, as the engine does.

## 14. Open questions

- Prefill/decode interface separation beyond shape dispatch — circle
  back for SSM/linear-attention blocks (chunkwise-parallel prefill vs
  recurrent decode) and speculative decoding (short-prefill-with-
  rollbacks; likely a new interface).
- Device transient sharing / scratch pool (first instance landed: the
  shared v4 prefill partials buffer, loader-owned, §9; generalize under
  memory pressure or when a second sharer appears).
- Trunk/norm merge: revisit triggers in §8.2.
- Naming: LmHead ("de-embedder" candidate; `lm_head` matches files).

## 15. Kernel selection framework (2026-09, foundation of the next-gen loader)

A kernel choice is pinpointed by, in order of when the information
becomes available:

- **Block type**, including dtype and quantization strategy.
- **Model** (per-model shapes; 1B vs 70B can warrant different kernels).
- **Hardware SKU**, not just ISA: same-arch parts differ in CU count,
  clocks, L2; winners can flip between SKUs (MI210 vs MI250X).
- **Parallel configuration** (TP degree): cuts matrices differently and
  adds the collective itself as a selection point (hand-rolled P2P vs
  overlay-RCCL, keyed by degree and message-size class; measurements in
  research/rccl-single-process/).
- **Call-time shape vector**: today `(S, T)` with T collapsed by
  measurement (v2 wins at all T); batching or speculative decoding will
  add B. The schema carries the full vector even when clauses ignore
  parts of it.

Deliberately excluded: **data-dependent dispatch** (e.g. MoE routing
makes shapes value-dependent). The accepted mask-and-allreduce EP
scheme keeps the dispatch closed — fixed expert GEMMs, masked.

The selection pipeline narrows the candidate set at four times:

1. **Code time** (architecture design): cull provably-inferior
   implementations; a human act backed by harness evidence. Case study:
   v1 attention retired after measurement falsified both of its
   dispatch boundaries. Second case study (2026-09): at S==1 the
   rocBLAS gemm_ex path was replaced by a family-private split-K GEMV
   in gpu_rocm — measured 26.0 of 29.6 ms/token at 313–1143 GB/s per
   shape vs a 1290 GB/s streaming ceiling; the GEMV lands 933–1276 GB/s
   and the block dispatch clause is `S == 1 → GEMV, else gemm_ex`
   (decode-roofline study: replica harness linking the engine's real
   kernel objects, per-shape A/B, CTA-target sweep).
2. **Compile time**: per-arch implementation TUs. Deferrable to load
   time via fat binaries (`--offload-arch` / `-gencode` lists; the
   driver picks the code object) so one package can serve all
   platforms; the cost is the per-arch impl families we already accept.
3. **Load time**: model + hardware + TP + config decisively known.
   Narrow to a one-dimensional table over the call-time shape vector.
   The loader orchestrates (.model descriptor + EngineConfig + platform
   probe → key), constructors receive their table, `forward()` does an
   O(1)
   dispatch (ideal: one integer compare, e.g. `S > 4`).
4. **Call time**: index the table; nothing else.

Norms (both already emergent practice, codified here):

- **Every dispatch clause cites a measurement.** Unmeasured entries are
  allowed for bootstrapping but must be marked provisional.
- **No entry = loud failure**, never silent substitution. A table miss
  means "this configuration was never validated."

**Measured-decision cache.** The tables are filled empirically, not by
reasoning: key = (model, device SKU, TP, software versions — kernel
source + compiler + flags, since codegen shifts flip winners; the
`__launch_bounds__` spill episode is the evidence), value = the
narrowed table plus the benchmark numbers that produced it. First load
on a machine may run a seconds-long autotune pass over a *closed,
precompiled* candidate set (no JIT, no runtime SDK — our answer to
FlashInfer's precomputed cubin matrix); subsequent loads read the
cache.

**Capacity is a fit check, not a selection dimension** (user decision,
2026-09). EngineConfig values (`max_seq_len`, `max_prefill_chunk`)
never affect which kernel is right — they only size buffers, and a cap
on S never changes a clause over the S axis. Since all allocation is
load-time anyway, the loader computes the complete VRAM bill (weights +
trunks + KV caches + scratch + lm_head, linear in the config knobs)
before the first allocation and compares against free memory — per-GCD
under TP, with a margin — failing with a message that names the knobs
and the largest fitting `max_seq_len`. Explicitly NOT done: silently
swapping to a leaner kernel under memory pressure — that would be an
unmeasured silent regression; if a model cannot fit at any reasonable
config, a leaner dispatch entry is a deliberate user-signed decision,
not a runtime adaptation.

## 16. S=1 HIP graph replay (2026-10)

One graph per rank records the block loop, final norm, and language head;
token upload/embedding and logits download remain outside capture, ordered
on the same explicit rank stream. The language head is warmed separately
before capture (never warm a full rank while its peers are idle at an
allreduce). No allocations occur during replay. Graph creation failure is
loud, not an eager fallback. `ModelConfig::decode_graph=false` is the
explicit eager A/B baseline; the REPL exposes `ALICE_DECODE_GRAPH=0`.

Replay parameters are instance-owned device allocations, not process-wide
constant symbols. Each attention block has a private device position cursor:
prefill RoPE publishes the next position, decode reads the cursor, and the
output-projection combine increments it without an extra launch. A private
host mirror validates context limits / supports subsequent eager prefill;
`StatefulBlock::advance_decode_state()` advances that mirror before replay.
Recording executes host bookkeeping but not kernels, so initialization
restores cursors after capture. Reset clears both position mirrors, but never
resets collective epochs.

Decode attention has a fixed physical grid of 64 splits per local query
head. Kernels read the current device position to derive active splits and
rows per split, preserving the old logical partition: 64 rows/split through
T=4096, then ceil(T/64). Partial stride is always 64; inactive split blocks
return and combine excludes their unwritten entries. This avoids per-token
graph updates/recapture. The overlaunch cost at short contexts is provisional;
smaller-capacity graph variants remain a measured future optimization.

Captured native decode collectives use immutable per-block slab offsets and static
collective ordinals plus a device epoch base uploaded once before each
replay. Host epochs advance by the block count; eager prefill and graph
replay therefore share one monotonically increasing sequence. Flags are
never cleared per replay (an independently cleared peer could still expose
stale readiness). Only the first produced barrier remains in each collective.

Execution initially uses regular blocking streams for a conservative
migration. Load-time repack uploads and kernels nevertheless explicitly use
the same rank stream and synchronize before scoped staging release, fixing
the previous nonblocking-stream staging race rather than relying on implicit
ordering. Input/output copies and replay completion also name the rank stream.

Validation includes one captured attention graph replayed across T=64/128
and 4096 boundaries for hd=64/128 with NaN-poisoned inactive scratch, plus
TP=2 graph/eager logits comparison through changing tokens, prefill after
replay, reset, and context overflow. CanonicalEnv supports optional TP and
decode-graph config (`ALICE_CANONICAL_{TP,DECODE_GRAPH}`).

Initial A/B (Release, `numactl --cpunodebind=0 --membind=0`, 70B, 28-token
prompt + 128 decode steps, attention context T=29..156, max_seq=8192):

| TP | Eager decode | Graph decode |
|---|---|---|
| 4 | 45.22 ms/tok, 22.1 tok/s | 44.29 ms/tok, 22.6 tok/s |
| 8 | 36.15 ms/tok, 27.7 tok/s | 34.91 ms/tok, 28.6 tok/s |

One run per configuration, not a variance-controlled sweep; all four
128-token generated texts were identical. Logs are in
`temp/graph-decode-ab/` (not committed). 70B S=1 canonical replay passed at
TP=4 (worst max drift 0.4951) and TP=8 (0.5040), gate 0.75. Full Debug
suite: 35 passed + one optional CanonicalEnv skip; targeted Release replay
and long-context tests passed three consecutive runs. The fixed-64 grid
is a correctness-first baseline; overlaunch at short T can offset some
host-submission savings.

## 17. Hybrid TP collectives: integration and validation (2026-10)

User-directed integration after standalone research commit ba8eb92:

- S=1 / <=16 KiB: retain the existing fused single-CTA ready+gather kernel.
  The initial extrapolation of the larger row's two-launch strategy regressed
  1B decode; the small-row path is not silently replaced.
- S=1 / 32 KiB: produced flag barrier + pointer-hoisted/dephased data-major
  gather, 2×1024. Contiguous CTA chunks; 64 B LDS peer-address cache; no
  payload prefetch or register-heavy peer-major tiling in production.
- S>1: out-of-place `ncclAllReduce` from the immutable block slot to trunk io,
  using one rank-owned prefill communicator on the rank stream. No added
  per-block host synchronization. The all-worker forward completion boundary
  still protects source reuse.
- Host collective counters advance for RCCL as well as native calls; RCCL
  leaves P2P flags alone. Next replay uses a higher base, so gaps are safe.
  Reset clears attention state, never collective counters.
- CMake links `ALICE_RCCL_LIBRARY` (default: the documented existing AMD 7.1
  overlay), with executable build RUNPATH. A symbol-path guard rejects a
  different loaded library before worker/communicator initialization. Version
  equality cannot detect Canonical's known launch-path bug. UID lives only in
  the load-phase ThreadState; owner workers call `ncclCommInitRank` after load.
  Decoder graphs are destroyed before communicator teardown, with rank device
  binding restored for cleanup. No initialization WorkItem was reintroduced.

Dephased FP32 addition order can differ across ranks. It is not a bitwise
replica claim: every implementation must pass the signed numerical gates.
1B TP=2, 70B TP=4 and TP=8 pass committed canonical cases for eager prefill
(chunk=512) and S=1 graph replay (chunk=1). Worst 70B drift is 0.547702,
below 0.75; 1B max is 0.356657. Existing graph/eager, reset, overflow, and
chunked-generation tests cover switching between vendor/native collectives.
The runtime wrong-library guard is tested with an explicit system-library
LD override and fails safely before a collective.

### Matched engine A/B

Baseline ba8eb92 built separately with the same Release flags; new hybrid
built in build-release. Public Engine API driver loads once, validates all
three 70B fixtures in both chunk modes, then generates 128 tokens per round.
No streaming callback or per-token text printing. CPU and memory are bound
with `numactl --cpunodebind=0 --membind=0`; max_seq=8192, chunk=512.
Short prompt is 28 tokens (attention T=29..156), long prompt 1667 tokens
(T=1668..1795). Short table uses the second warm round; these are matched
initial runs, not a variance-controlled campaign.

| TP | short decode baseline → hybrid | long decode baseline → hybrid | 1667-token prefill baseline → hybrid |
|---|---|---|---|
| 4 | 44.72 → 44.18 ms/tok (22.36 → 22.63 tok/s) | 46.89 → 46.59 ms/tok (21.33 → 21.46 tok/s) | 1.853 → 1.449 s (900 → 1150 tok/s) |
| 8 | 34.99 → 33.20 ms/tok (28.58 → 30.12 tok/s) | 36.76 → 34.92 ms/tok (27.20 → 28.64 tok/s) | 2.244 → 0.917 s (743 → 1818 tok/s) |

TP=8 long prefill improves ~2.45×. All short generated texts match the old
binary; long TP=4 matches, long TP=8 differs. The committed teacher-forced
canonical gate passes; no fp32-ideal long-70B greedy-margin fixture was used,
so do not claim exact long greedy equivalence. Small-model results are not
universally faster: final 1B TP=2 warm short decode 246.6 → 239.6 tok/s,
long 219.1 → 213.8, despite retaining its fused native decode. Vendor prefill
and communicator resources are part of this hybrid; the small regression's
precise cause is not isolated.

Logs, the temporary public-API driver, and the detached baseline worktree are
under temp/hybrid-collectives/ and temp/hybrid-baseline/ (not committed).

## Retired decisions (kept so they are not re-litigated)

- **The CPU production path** (host Model/Trunk/components, CPU_AVX2
  kernels, host-io GPU FFN + hipRTC, HardwareClass factories): deleted
  2026-09 after the GPU-resident path passed the full gate suite,
  including fp32-ideal long-case validation. In git history. The
  `cpu_generic` host-block reference impls are retired too — naive
  reference impls for new block types are now done as GPU kernels; the
  host-side math oracle survives as `src/host_math`. Note: the 70B
  (141 GB > one GCD) is intentionally dark until multi-die placement
  lands (§9).
- **CPU-first bring-up** with hand-rolled AVX2 kernels and NUMA tuning:
  served bring-up well (matched OpenBLAS sgemm at 1.2 TFLOP/s); retired
  with CPU-only builds. In git history.
- **No-BLAS verdict**: scoped to CPU BLAS on Zen3; stands there and is
  irrelevant to the device path (vendor BLAS is used where it fits).
- **hipRTC string kernels**: a workaround for avoiding hipcc; retired.
- **Hardware-class fallback chains** ending in a CPU oracle: replaced by
  per-arch selection with loud failure. (The dev-only CPU reference blocks
  that sat alongside are retired — see §3.)
- **FP32 host io boundary**: the fp32 boundary moved to the device;
  precision unchanged, address space changed.
