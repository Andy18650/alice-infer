# Tile-local last-arriver reduction

User proposal: publish a split's partials, atomically decrement its counter,
and let the CTA observing old count one reduce immediately. **Implemented and
validated**, but the initial net-cost comparisons do not justify default
adoption. The two-kernel path is retained unchanged as a control.

## State and algorithm

Each output-row tile (512 rows) has one uint32 counter initialized to the
number of K splits. The selected split-specific geometry and scratch belong
to one serialized stream; they are **not reentrant across concurrent streams**.

For every job, a CTA:

1. Computes exactly the same two-row-per-thread FP32 partials as the ordinary
   producer and stores them in `[split, padded_rows]`.
2. Publishes **all writing threads'** stores with a device/agent-scope fence,
   then joins a CTA barrier. Only thread zero modifies the tile's counter.
3. Atomically adds uint32(-1), obtaining the old count. This is subtraction
   modulo 2^32 using HIP's supported atomic-fetch-add builtin.
4. Broadcasts `old == 1` through a 4-byte shared flag and CTA barrier.
5. A non-last CTA **continues its grid-stride job loop**, rather than returning
   and abandoning future jobs. It never spins or waits for another CTA.
6. The winner acquires published partials and each thread reduces its own
   pair of output rows in fixed split order, writing FP32 y.
7. After a CTA barrier, thread zero performs an agent-scope release atomic
   store restoring the counter to `splits`; another CTA barrier protects
   shared-flag reuse. Ordered kernel completion protects y and scratch reuse
   before the next same-stream operation.

Counter zero is an **arrival state**, not an externally consumable y-ready
signal: reduction has not yet necessarily finished. Consumers wait for
kernel completion. Reset does not permit overlap with a second stream.

No per-operation initialization kernel is timed or required. Initialization
happens in candidate/load setup; every completed fused operation restores
its own state. Changing geometry requires setup before using it. Unsplit
GEMV dispatches the ordinary direct-output kernel with no counter/fences.

The last-CTA scheme's progress comes from **not waiting**, not a 104-block
residency assumption. Natural grids can be oversubscribed safely. Tests
include 2050 CTAs and one CTA serially handling all 2050 jobs.

## Two ordering builds

`build.sh` retains two versions:

- `bench-strong`: every producer uses `__threadfence()` (agent seq-cst),
  followed by an agent acq-rel counter RMW. The winner also performs an
  acquire fence before reading partials.
- Default `bench`: every writer uses an agent **release** fence before the
  CTA publication barrier. The elected thread uses a **release** RMW.
  If its returned old value is one, it executes an acquire fence **before
  broadcasting the winner flag**. Winner threads also acquire before loads.

The optimized protocol uses atomic-fence synchronization: the old value
read by the last RMW comes from the preceding unbroken release/RMW sequence.
The winner's acquire fence follows that atomic read and synchronizes with
the preceding release contributions; CTA synchronization carries this to
other winner threads. The writers' publication and leader release must not
be replaced by an ordinary relaxed decrement without ordering. Scope is
one GPU agent, not just wave or workgroup, and not a peer/system protocol.

Generated code records one returned-value global atomic add and four CTA
barrier sites, complete producer weight batches, and no scratch or register
spills. Strong ordering emits three vector-cache-invalidation sites; the
release version emits two, confined to the winning path. These are static
sites, not a measurement of how many execute per invocation.

## Counter placement control

Benchmark both adjacent 4-byte counters and one counter per 256-byte slot.
Separate state arrays have the same lifecycle; padding is not interpreted
as a proven hardware channel map, and base addresses also differ. Neither
counter layout consistently wins. All initialization and state checks are
outside the timed operation; self-restoration remains inside.

## Correctness

Every candidate and layout is tested against the FP64 oracle and **bitwise
against the two-kernel reduction**. The fused row-pair reducer preserves
its eight-accumulator grouping and arithmetic tree; its leftover split
loads are prefetched together rather than serially issued.

Existing odd N/K, empty K, padding, activation-precision, zero/cancellation
and packing tests remain. Additional stress:

- Sixteen back-to-back same-stream invocations, no host counter reset.
- Negated FP32 activations and poisoned partial/y buffers, while retaining
  the same weights and counters; repeat again and check all real outputs.
- Restore original activations without resetting counters and validate.
- Both compact/spaced state, both publication builds.
- Oversubscribed natural grid: N=513, K=259, splits=1025, jobs/grid=2050;
  also the same geometry with grid=1. Neither can rely on global residency.
- Read counters after completion and require every tile restored to splits.

The release and strong tests pass; timed batches also validate output and
counter restoration. No floating-point atomic accumulation or arrival-order
sum is used, so changing the last CTA does not change the reduction order.

## Net measurements

Native O3 HIP/C++20, gfx90a, HIP GPU 0 / PCI 0000:31:00.0, CPU+memory NUMA
node 0, no model/context, no clock overrides. Same allocations, geometry,
weight format and FP32 inputs within each process. Nine shuffled rounds,
two process repeats for each final protocol; phase-only scheduling barriers.
Sampled active graphics/memory clocks 1700/1600 MHz. Useful weight-byte
rates are not actual HBM traffic. GPU event intervals include all work;
CPU wall includes submission and final wait, excludes validation/setup.

Fixed gup N14336 K8192: interleaved, batch8, splits3, 84 blocks/jobs.
Fixed o N2048 K8192: interleaved, batch8, splits26, 104 blocks/jobs.

Release-publication build, **GPU useful-byte GB/s**, twelve operations/batch:

| Shape | Two kernels | Last-arriver compact | Last-arriver spaced |
|---|---:|---:|---:|
| gup | 1374.4 / 1393.5 | 1386.4 / 1387.0 | 1373.1 / 1373.0 |
| o | 1023.8 / 1022.6 | 1010.3 / 1000.2 | 993.9 / 992.3 |

Paired CPU wall savings versus two kernels tell the same story: gup compact
wins 5/9 and 4/9 rounds, median saved +0.35/-1.35 us. o compact wins 0/9
in both processes, median losses 0.49/0.83 us. Counter spacing does not
rescue these configurations.

**Single-operation CPU wall us**, two independent nine-round processes:

| Shape | Two kernels | Last-arriver compact | Last-arriver spaced |
|---|---:|---:|---:|
| gup | 181.83 / 180.91 | 181.71 / 181.17 | 182.59 / 184.17 |
| o | 48.04 / 47.67 | 49.48 / 49.33 | 49.51 / 49.92 |

Again, do not infer a gain from the difference of two process medians:
gup compact wins 4/9 and 3/9 paired rounds, median saved -0.61/-1.19 us.
o compact wins 1/9 and 0/9, median saved -1.34/-1.48 us. Removing a launch
has not overcome publication/atomic/CTA-reduction costs in these cases.
The stronger first implementation was also mostly slower; its reports
remain separately retained.

This does not prove last-arriver reduction is inherently inferior. Producer
register allocation changes (winning interleaved/batch8 fused reports 54
registers and 4 B LDS versus ordinary 58 registers/zero LDS); reduction work
is tile-local and maps two outputs per thread rather than the control's one.
Fence and contention attribution is not isolated by hardware counters.
No baseline promotion or production change follows from these results.

## Reproduce / artifacts

```sh
research/gemv-layout/build.sh
research/gemv-layout/run.sh temp/gemv-layout/new-last quick 9 12 release
research/gemv-layout/run.sh temp/gemv-layout/new-last-latency quick 9 1 release
research/gemv-layout/run.sh temp/gemv-layout/new-last-test test 3 1 strong
```

New phases: `gemv_last`, `gemv_last_spaced`; `gemv_full` remains the ordinary
producer + combine. `fused-resources.csv` reports LDS/register capacity;
`check-fused.py` saves all six kernel bodies and checks atomic/barrier and
weight-batch emission before timing. Setup owns counter initialization;
launching last-arriver itself never reinitializes state.

Curated: `results/last-*.csv`, `last-arriver-paired.csv`, and per-publication
ISA/resource reports. Full snapshots, raw rounds, GPU code objects and
telemetry: corresponding `temp/gemv-layout/` folders. Initial `last-quick-*`
and `last-latency-a` precede the publication-policy selector and use the
strong protocol; release reports are explicitly named `last-release-*`.
These are synthetic weights/model-shaped matrices, not full engine runs.
