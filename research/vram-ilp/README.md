# Streaming reads with 104 × 256 threads: source ILP vs generated scheduling

User-directed follow-up to `vram-ceiling`: can one CTA per CU approach the
read ceiling using independent loads within each thread? Why did unroll 8
reach ~1460 GB/s while unroll 16 collapsed to ~550 in the earlier screen?

## Initial research checkpoint

**The unroll-16 regression is explained by generated load scheduling, not
an occupancy loss or a demonstrated channel conflict.** Holding addresses
and arithmetic fixed, constraining scheduling restores about **2.5×**
throughput. Wave-local dense tiles improve further. 104×256 reaches a
nine-round 1 GiB median of **1502.9 GB/s in one run**, but the independent
repeat of that exact configuration is **1486.4**; do not call 1500+ a stable
ceiling yet. Repeated wave-tile results are broadly ~1470–1495 at 1 GiB,
~1440–1460 at 2 GiB, and ~1420 at 4 GiB.

Conditions for all rates/timings here: standalone native **O3 HIP/C++20**,
gfx90a, HIP logical device 0, **PCI 0000:31:00.0, 104 CUs**, CPU+memory
NUMA node 0 via numactl, **no model/context**. Decimal GB/s counts input
bytes only, not memory-controller traffic. HIP 7.1.52801, Ubuntu clang
21.1.8; no clock/power overrides. Final event-timed runs sample 1700/1600
MHz core/memory, memory temperatures about 53–71 C. Timings include wave
checksum reduction and sparse output stores, but exclude initialization,
allocation and host checksum readback.

### 1. Unroll does not specify outstanding loads

The preserved executable's actual `read_stream<2,8,false>` code issues
eight `global_load_dwordx2` instructions before consuming results, then
uses graduated waits. Its unroll-16 counterpart instead reuses one load
result register pair and drains VMEM after each load:

```text
# U=8, schematic hot loop
load v0
load v1
... eight loads ...
wait vmcnt(6)
consume a completed result
wait vmcnt(5)
...

# U=16, schematic hot loop, repeated sixteen times
load temporary
wait vmcnt(0)
accumulate temporary
```

Both have independent accumulators in C++. That did not prevent the
backend from emitting serialized memory requests for U=16. U=8 reports
64 registers, U=16 only 59; **both have the same 8-CTA/CU capacity** at
256 threads. The regression is not a drop in the occupancy limit.

The new probe loads the preserved gfx90a object through the HIP module
API and benchmarks its kernels alongside new candidates. This avoids a
silent compiler-version/source-change explanation for the old result.
New ordinary global U=8/U=16 kernels reproduce the old codegen/resource
counts and track the preserved kernels' rates in matched runs.

### 2. A controlled scheduling change restores throughput

Three policies share the same addresses, input, checksum arithmetic and
launch geometry:

- `immediate`: load a vector and accumulate it, as in the original source;
- `staged_cpp`: load into a temporary array, then consume the array;
- `staged_sched`: the same staged array with
  `__builtin_amdgcn_sched_barrier(0)` between load/consume phases.

The intrinsic is an AMD compiler scheduling constraint, **not a GPU memory
fence, CTA barrier, grid barrier, or host synchronization**. It can change
register allocation, instruction order and inserted nops; it is not a
claim of zero performance overhead.

Pure C++ staging still generates serialized U=16 loads. The scheduling
constraint preserves all sixteen loads before the first consumption wait.
The checked variant reports 90 registers and zero scratch, versus 59 for
the ordinary kernel. Its occupancy capacity is 5 CTAs/CU instead of 8,
but the requested grid contains only 104 CTAs.

Matched offset-zero medians from `pingpong-{a,b}`, nine rounds × 12 reads:

| Global traversal, 104×256 | 1 GiB A/B | 2 GiB A/B | 4 GiB A/B |
|---|---:|---:|---:|
| U=16, ordinary | 558.6 / 558.8 | 559.8 / 559.7 | 560.3 / 560.4 |
| U=16, staged scheduling | **1390.6 / 1395.8** | **1384.1 / 1384.6** | **1383.4 / 1383.3** |

The staged kernel retains `i = global_thread + k*total_threads`; no layout,
stride, data-volume or block-count change explains this improvement.
The codegen intervention also changes resource allocation, so it is not a
hardware-counter measurement of a uniquely isolated request-queue limit.

### 3. Dense wave tiles outperform whole-grid interleaving here

The layouts all read every input word exactly once per invocation:

- `global`: next per-thread load is one whole grid apart;
- `wave`: each wave owns U contiguous 512-byte strips, then advances by a
  whole grid of those tiles;
- `CTA`: each CTA owns U contiguous 2048-byte strips;
- `slab`: one contiguous partition per CTA (partition boundaries in vector
  units, not rounded to cache/transaction boundaries).

For `wave`, U=8, lane l of wave w starts at vector index
`w*64*8+l`, reads offsets `u*64`, and advances by
`total_threads*8`. A wave therefore reads one dense **4 KiB tile** in eight
coalesced 8-byte-per-lane loads. Adjacent waves own adjacent 4 KiB tiles.
This preserves full dense coverage, but **not the original concurrent
address distribution**. No unused padding or weight repacking is involved.

Selected offset-zero confirmation medians, same final matched runs:

| Kernel | Geometry | 1 GiB A/B | 2 GiB A/B | 4 GiB A/B |
|---|---|---:|---:|---:|
| Historical global U=16, unchanged code object | 832×256 | 1404.7 / 1398.9 | 1391.7 / 1391.5 | 1387.4 / 1387.1 |
| Global U=16, scheduling constrained | 104×256 | 1390.6 / 1395.8 | 1384.1 / 1384.6 | 1383.4 / 1383.3 |
| Wave U=8, one-step source pipeline | 104×256 | **1484.1 / 1483.4** | **1449.2 / 1437.2** | **1416.5 / 1418.6** |

Thus 104 CTAs can match the contemporaneous 832-CTA global control while
retaining global addressing, and exceed it with a wave-tiled address
schedule. **The unchanged control no longer reproduces its historical
1530.5 median in this campaign.** The cause of that cross-campaign shift
is not isolated. Compare matched controls, not historical maxima, when
attributing an improvement to a kernel change.

The later `marked` confirmation mixes a timestamped candidate and the
same uninstrumented controls. Uninstrumented wave U=8 pipeline, offset
65536, 1 GiB: A median **1486.4** (1477.0–1505.8), B **1502.9**
(1463.8–1522.3). Individual batches above 1500 are common; the stable
independent-repeat claim remains roughly 1.48 TB/s, not 1.53.

### 4. Bigger buffers, unrolls and register banks are not automatic wins

Exploratory sweeps include U=1/4/8/16/32/64, four address schedules, three
load policies, and a separate `__launch_bounds__(256,1)` codegen axis.
Without the bound, several larger variants spill to scratch; these are
recorded as `excluded_scratch` and never timed as clean streaming reads.
The bound permits more register use for a 256-thread CTA and removes
those spills in tested large variants, but **does not consistently improve
throughput**. `reported_regs` is HIP's resource attribute, not a statement
that every reported register is an ordinary VGPR on gfx90a.

We also tried:

- `pipeline`: current/next source-level arrays, issuing the next tile before
  accumulating the current one;
- `pingpong`: two explicit alternating register banks, consuming/refilling
  A while B is available and then consuming/refilling B.

The one-step pipeline's generated register-copy waits can drain its loads
at a loop boundary: do not assume source buffer count implies twice as
many requests outstanding. The explicit ping-pong candidate also failed
to improve the best result consistently (wave U=8 ~1396 at 1 GiB and
~1392 at 4 GiB in final runs). U=16 is not categorically bad: CTA-tiled
ordinary U=16 reaches ~1460 at 1 GiB. The bad case is a particular codegen
and address schedule, not the integer sixteen itself.

### 5. CU placement is checked, not inferred from grid size

A private timestamped twin of wave U=8 staged scheduling records CU IDs
and per-wave entry/exit steady clocks. No artificial holds, lane/CTA
barriers, atomics or cooperative launch. It reports 40 registers, zero
scratch and the same 8-CTA/CU capacity (ordinary candidate reports 42).

Seven 1 GiB samples all show:

- **104 CUs seen, exactly one CTA assigned to each**;
- all 416 wave intervals overlap;
- 0.16–0.32 us wave entry spread, common overlap about 685 us median.

That is **four resident waves/CU**, 12.5% of the reported 32-wave maximum
in the marked overlap window, despite high read throughput. It is not an
average hardware occupancy counter for the uninstrumented kernel.
Do not divide the whole input size by the common-overlap interval to infer
bandwidth: that interval ends when the *first* wave finishes, before the
whole input read completes. Marked bandwidth is measured separately with
HIP events; it does not supply a consistently faster winner.

## Protocol, codegen and artifacts

`probe.hip` is standalone O3 native HIP/C++20 with no engine headers,
objects or rocBLAS. One ordinary hipMalloc input of 4 GiB+64 KiB, deterministic
nonzero dispersed integer initialization, separate sinks/markers. Input
reads use uint2; checksum is a host-verified modulo-2^32 sum over **all**
wave outputs, with analytic adjustment for the input base offset.

Small/tail controls: 64 B, 4160 B and 1 MiB+64 B. Each timed batch also
passes its full-payload checksum. No allocations or transfers inside timing.
No explicit cache flush: 1/2/4 GiB greatly exceed L2. Values are not a model
or GEMV workload, and input-byte GB/s is not actual HBM transaction GB/s.

- Broad sweeps: layouts × U × source policy × launch-bound axis, plus
  preserved-code-object controls. Resource queries record excluded scratch
  variants before timing. `eligible` means resource-eligible, not necessarily
  selected/timed by a narrowed phase.
- Focus confirmation: selected closed candidates, three sizes and input
  offsets 0/256/8192/65536, **jointly shuffled** each round; 250 dense warmup
  reads, two case warmups, nine rounds with 12 reads/batch. Independent
  process runs use fresh allocations and preserved source/binary hashes.
- `marked` phase: six preselected kernels, same sizes/offsets/protocol, to
  check the timestamped candidate with proper whole-kernel event timing.
- `trace` phase: seven single-launch timestamp/CU traces; not bandwidth data.

Actual executable disassembly is retained under
`temp/vram-ilp/marked-a/disassembly.txt`, with extracted selected functions.
All checked winning/root-cause variants retain 8-byte loads and have no
scratch instructions. Original disassembly and U=4/8/16 excerpts are under
`temp/vram-ilp/original-*`. Whole-function opcode counts include prologue
and tail loads; they are not per-iteration request counts.

Final summaries: `results/pingpong-{a,b}.csv`, `marked-{a,b}.csv`,
`resources.csv`, `trace.csv`. Full raw batches, original code object,
preserved candidate source/binary, hashes, device/clock/temperature data
and traces: corresponding `temp/vram-ilp/` directories. Earlier broad,
bounded and focus runs remain there as exploratory artifacts.

## Reproduce

```sh
research/vram-ilp/build.sh
research/vram-ilp/run.sh temp/vram-ilp/new-all temp/vram-ceiling/final-1/ceiling 7 all
research/vram-ilp/run.sh temp/vram-ilp/new-focus temp/vram-ceiling/final-1/ceiling 9 focus
research/vram-ilp/run.sh temp/vram-ilp/new-marked temp/vram-ceiling/final-1/ceiling 9 marked
research/vram-ilp/run.sh temp/vram-ilp/new-trace temp/vram-ceiling/final-1/ceiling 9 trace
```

The second argument is a preserved **ceiling executable**, not a raw object;
the runner extracts its embedded gfx90a object. It must contain the original
read_stream specializations. `all` can take substantially longer than the
narrowed phases. Scripts fail on stale candidate binaries or bad checksums.
Python helpers use only the standard library and reuse the ceiling study's
read-only telemetry helper. No package/system changes.

## Implications and next discriminator

- A source-level unroll/independent-accumulator count is not an MLP measure.
- High memory throughput does not require maximum resident-wave occupancy.
- Keep addressing and code scheduling as separate measured axes.
- Larger register budgets are useful for avoiding spills, not a prescription
  to maximize unroll or sacrifice occupancy indiscriminately.
- Counter collection is still blocked by the previously recorded AQL-profile
  extension failure. It is needed to quantify request queues, cache traffic
  and downstream imbalance; ISA alone is not a counter substitute.
- Applying these ideas to GEMV needs its own tests: quantization, FMA work,
  weight/activation traffic and reductions change the scheduling problem.

This checkpoint changes no production kernel, layout or dispatch policy.
