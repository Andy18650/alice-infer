# A / B / C: coalescing per instruction versus eventual byte coverage

**Research checkpoint:** user-directed kernel-loop exploration is closed for
now. Several coalesced, sufficiently batched readers sustain roughly
1350–1450 GB/s, with shape/size-dependent excursions. The next study will
focus on VRAM layout constraints in practical kernels. This is a research
priority, not proof of one hardware bottleneck or a production dispatch rule.

User-directed comparison. The current baseline uses block/thread load
indexing: one CTA owns `4*blockDim.x` dwords (4 KiB at 256 threads), and
every reader covers that region once. It uses raw `uint32_t` input, HIP's
built-in `uint4` and an exact modulo-2^32 checksum. There are no float
bitcasts or custom compiler-vector types. See the [indexing comparison](indexing.md)
for the measured equivalence and the separate historical wave control.
The [smaller-grid sweep](small-grid.md) finds 80 blocks retain near-full
bandwidth, 64 begin losing it, and 32 fall sharply, with A/B otherwise fixed.
The [larger-window follow-up](inflight.md) recovers near-full throughput
for A at 64 blocks/eight chunks; B's sixteen-chunk code spills SGPRs into
VGPR lanes despite retaining full load batches and no scratch traffic.
The [chunk-order comparison](chunk-order.md) gives each CTA adjacent tiles:
A/four chunks at 64 blocks improves, larger A batches often lose, and
B/sixteen chunks becomes much faster as the hot SGPR-lane spill code disappears.
The [complete order matrix](order-matrix.md) covers A/B at all 104/80/64
blocks × 4/8/16 chunks × both orderings, with matched 1/2/4 GiB results.
The [window/warmup study](windows.md) reads each 1 GiB quarter of one
4 GiB allocation, monitors firmware average/current clocks, and finds no
warmup-linked bandwidth decline or consistently bad quarter. FCLK reporting
remains ambiguous; absence of a fabric-clock change is not proven.
The [four-body control](body-control.md) compares one 4 GiB traversal,
four body calls in one grid, and four separate kernels, including GPU and
CPU wall time. A's streaming gain is small; single-operation latency wins
are irregular, and the multi-body wrapper changes register allocation.
The current default has no compiler scheduling barriers; see the
[barrier-removal and block-count follow-ups](tuning.md) for matched controls.
The [rotating register-buffer experiment](ring.md) adds a depth-four ordered
B ring with a small repeatable gain at 104×256; A does not improve.

## Raw-integer simplification and revalidation

The initial comparison used float-bit data and Clang vector aliases. Those
historical tables and snapshots below remain intact; they are not relabeled
as measurements of the new implementation. The integer rewrite uses
`p[i] = uint32_t(i)*1664525 + 1013904223` and an analytic triangular-sum
checksum, removing the float/mantissa initialization and conversion helpers.

Raw-integer conversion checkpoint with scheduling barriers enabled: one
process per build, nine shuffled rounds × 12 reads, 1 GiB, 104×256,
chunks=4, same device/placement conditions as below:

| Reader | Controlled build | Normal O3 build |
|---|---:|---:|
| A, built-in uint4 | 1428.9 | 1454.4 |
| B, striped uint32_t | 1427.4 | 1433.8 |
| C, contiguous uint32_t | 674.6 | 1454.8 |

All short/tail and full-payload checks pass across the 1/2/4 GiB sweep.
At chunks=4, A retains 16-byte loads; controlled B/C remain scalar; normal
C widens to 16-byte loads. At chunks=1 in the controlled build, HIP `uint4`
lowers to **two 8-byte loads**, not one 16-byte load. The checker records
actual widths and permits this explicit split instead of forcing a custom
vector type. A logical vector type is not a hardware instruction guarantee.

A seven-sample packing rerun preserves the previous 1/2/4/8 CTA-per-CU
spreading at grids 104/208/416/832. These new summaries are named `uint-*`
in `results/` and `temp/vram-coalescing/`. No historical data is overwritten.

## Original comparison results

**A and B perform similarly once there is enough independent work. Explicitly
scalar C is about half as fast and is not rescued by more concurrency.**
The normal compiler vectorizes C into A's wide-load pattern, removing the
penalty. This is a statement about generated instructions, not a blanket
warning against contiguous values per lane.

All numbers: native O3 HIP/C++20, gfx90a, HIP logical device 0,
**PCI 0000:31:00.0, 104 CUs**, CPU+memory NUMA node 0 via numactl,
**no model/context**. Input-byte decimal GB/s, not physical HBM traffic.
HIP/hipcc 7.1, Ubuntu clang 21.1.8. No clock/power changes; sampled active
clocks 1700/1600 MHz core/memory. Sampled memory temperatures in the
controlled repeats were about 56–68 C.

### Original wave-based accesses within one tile

For lane l and component k:

| Reader | Dword index | Intended hot memory instructions per tile |
|---|---|---|
| A, vector contiguous | `4*l + [0,1,2,3]` | one 16-byte vector load per lane |
| B, scalar striped | `l + 64*k` | four scalar loads, adjacent lanes adjacent dwords |
| C, scalar contiguous | `4*l + k` | four scalar loads, adjacent lanes 16 bytes apart |

A/C have the same per-lane ownership; B transposes ownership within the
wave. They preserve the global byte set, not necessarily any real algorithm's
per-lane result. Every full-tile load is aligned as required; base/size
alignment is held fixed. Tail reads are masked separately.

### Controlled comparison: intended widths verified

Two independent process runs, nine randomized rounds × 12 reads/batch,
104×256, 1 GiB. `chunks` is the number of independent tiles staged per
iteration, not the element/vector width:

| Chunks | A median A/B | B median A/B | C median A/B |
|---|---:|---:|---:|
| 1 | 957.6 / 954.6 | 929.7 / 930.3 | 652.4 / 652.7 |
| 2 | 1336.4 / 1369.6 | 1352.5 / 1349.4 | 676.8 / 676.8 |
| **4** | **1432.3 / 1427.2** | **1430.3 / 1434.6** | **675.6 / 675.6** |
| 8 | 1353.3 / 1353.8 | 1427.5 / 1435.5 | 675.8 / 675.6 |

At 4 GiB/chunks=4, A is 1398.0/1398.6, B 1392.1/1392.6,
C 677.2/677.2 GB/s. At 832×256, C stays about 674 at 1 GiB and 676
at 4 GiB across the chunk counts. Those larger grids are launch sizes,
not claims that every variant fits entirely resident at once; capacities
are recorded in `resources.csv`.

This is not the earlier compiler-serialization cliff: at chunks=4,
**B and C both issue sixteen scalar loads before the first wait** and have
the same graduated wait sequence, 52 reported registers, zero scratch,
and 8-CTA/CU occupancy capacity. A reports 50 registers and the same
capacity. No `glc`, `slc`, or `dlc` flag differences are present.

### Normal compiler: C is no longer scalar

Controlled mode applies these flags equally to all readers:

```
-fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false
```

The original A used a Clang vector type and retained `global_load_dwordx4`.
Current A uses HIP `uint4`; the actual 16-byte or split 8-byte loads are
reported explicitly. Controlled B/C retain scalar `global_load_dword`. No volatile loads or altered cache
policies are used. AMD scheduling constraints prevent the staged results
being consumed before the intended load batch; they are compiler controls,
not GPU synchronization. Earlier IR vectorization can still merge C's
source loads in the normal build, before those scheduling constraints act.

Two normal-O3 process repeats, chunks=4, 104×256, 1 GiB:

| Reader | Median run A | Median run B | Actual hot load width |
|---|---:|---:|---|
| A | 1426.0 | 1441.7 | 16 bytes |
| B | 1421.0 | 1442.7 | 4 bytes |
| C | **1432.1** | **1422.0** | **16 bytes** |

At 4 GiB all three are roughly 1390–1400. The scalar-width controls do not
materially change A/B's result; C recovers because its four adjacent loads
are combined into the wide instruction. The controlled build deliberately
measures a real scalar instruction pattern, not what this particular
normal compiler chooses for that source.

## What this establishes

- Eventually reading the same bytes does not imply equivalent throughput.
- Four coalesced scalar passes (B) can match one wide pass (A) with adequate
  batching; fewer instructions alone are not a universal throughput winner.
- Scalarizing per-lane contiguous records (C) produces inefficient accesses
  within each instruction: 256 useful bytes per wave instruction spread over
  a 1 KiB address span, versus B's contiguous 256-byte span.
- Any cache reuse between C's passes does not restore A/B's measured rate here.
- This supports a transaction/request/coalescing penalty, **not a measured
  fourfold increase in HBM traffic or a uniquely identified LSU/cache unit**.
  Hardware counters remain unavailable due to the previously recorded
  AQL-profile initialization failure. Do not infer a physical channel map.
- Simply doubling batching from four to eight can hurt A while B remains
  fairly steady. More unroll or wider loads is not automatically better.

## CU/SIMD packing

The follow-up [packing study](packing.md) records hardware CU, SIMD and
wave-slot IDs plus per-wave entry/exit times. A/B/C spread identically:
104 CTAs use all 104 CUs with one CTA/CU and one wave on each SIMD; grids
208/416/832 give 2/4/8 co-resident CTAs/CU. A 1664-CTA control queues later
work rather than making sixteen CTAs resident per CU. See the linked report
for repeated traces, resource checks and instrumentation caveats.

## Code and protocol

Three named readers (`A`, `B`, `C`) in private `readers.hpp` make their
accesses explicit, with one shared traversal/checksum used by the bandwidth
and packing probes. The baseline load loop uses only block/thread IDs;
wave arithmetic is confined to the final checksum and `wave-control.hpp`.
No W=1/2/4 element-width branch ladder. HIP's built-in `uint4` keeps the
four-component accumulation concise.

Scheduling constraints are opt-in. `RESEARCH_SCHED_BARRIERS=1` enables
load/consume phase barriers; `RESEARCH_LOAD_BARRIERS` defaults to the same
value and also constrains each load. Setting the latter to zero retains
only phase boundaries (`phase` runner option). These are compiler controls,
not GPU synchronization or memory-completion waits.

The baseline starts at `blockIdx.x * block_words`, where
`block_words=4*blockDim.x`. It stages `chunks` CTA regions separated by
`gridDim.x*block_words`, then advances by `chunks` times that stride.
A/C read `base+4*thread+component`; B reads
`base+thread+component*blockDim.x`. One unique checksum output per native
wave avoids races; the host checks the sum of all wave outputs.
Current initialization generates raw 32-bit words with an affine sequence;
the host checksum sums that sequence analytically with modulo-2^32
arithmetic. It is synthetic data, not a model or realistic arithmetic cost.

One ordinary hipMalloc 4 GiB input, initialized outside timing. Small/tail
checks now include 0/4/64/68 B, 4160 B, 1 MiB+64 B and four bytes below/
above a full staged-grid batch. Earlier reports used the original three
sizes. Full checksum after every timed batch.
No allocations or host transfers inside the event interval. Reduction and
sparse checksum writes are timed. No explicit cache flush; 1/2/4 GiB working
sets exceed L2. No host/io/model/attention activity inside the benchmark.

The quick phase measures A/B/C, chunks=1, 104×256, 1 GiB. The sweep uses
chunks=1/2/4/8, grids=104/832, block=256, sizes=1/2/4 GiB jointly shuffled
in each round. 250 preheat reads, two case warmups, nine rounds of twelve
reads. Two controlled and two normal compiler process repeats are retained.
No scratch-using candidate is accepted.

`check-loads.py` inspects load/wait sequences, cache flags and scratch
instructions **before** future runs. Barrier-on controls enforce expected
load shapes; barrier-free runs record scheduling and merging changes instead
of rejecting the changes under test. Cache/scratch checks remain strict.
It separates the first hot load batch from masked tail instructions, which
are scalar even for A. `load-widths.csv` records the actual widths for A
and C rather than assuming one uint4 expression means one instruction.
The checker was renamed to avoid shadowing Python's standard `inspect` module.
The recorded runs were also inspected retrospectively with this checker.

## Artifacts and reproduction

Summaries, load-width reports and resource tables are in `results/`.
Raw rounds, sources, executables, hashes, ISA, device information and
read-only telemetry are in `temp/vram-coalescing/{sweep-a,sweep-b,default-a,
default-b,quick-a}/` (gitignored, durable repository-local storage).

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-quick controlled quick 9 on
research/vram-coalescing/run.sh temp/vram-coalescing/new-sweep controlled sweep 9 on
research/vram-coalescing/run.sh temp/vram-coalescing/new-normal default sweep 9 off
research/vram-coalescing/run.sh temp/vram-coalescing/new-blocks default blocks 9 off
research/vram-coalescing/run.sh temp/vram-coalescing/new-index default index 9 off
research/vram-coalescing/run.sh temp/vram-coalescing/new-small-grid default small-grid 9 off
research/vram-coalescing/run.sh temp/vram-coalescing/new-inflight default inflight 9 on
research/vram-coalescing/run.sh temp/vram-coalescing/new-inflight-phase default inflight 9 phase
research/vram-coalescing/run.sh temp/vram-coalescing/new-order default order 9 phase
research/vram-coalescing/run.sh temp/vram-coalescing/new-order-all default order 9 on
research/vram-coalescing/run.sh temp/vram-coalescing/new-order-matrix default order-matrix 9 phase
research/vram-coalescing/run.sh temp/vram-coalescing/new-windows default windows 9 phase rich
research/vram-coalescing/run.sh temp/vram-coalescing/new-windows-basic default windows 9 phase basic
research/vram-coalescing/run.sh temp/vram-coalescing/new-body-control default body-control 9 phase rich 12
research/vram-coalescing/run.sh temp/vram-coalescing/new-body-latency default body-control 9 phase basic 1
```

All binaries use O3 C++20/gfx90a. The final argument selects scheduling
barriers off (default) or on (matched control). The runner rejects stale binaries,
verifies generated load shapes, records metadata and reuses the ceiling
study's read-only telemetry helper. Python is standard-library only.
No production engine, package or system changes.
