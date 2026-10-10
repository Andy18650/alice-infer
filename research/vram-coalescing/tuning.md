# Follow-ups: removing scheduling barriers and increasing block count

Raw uint32_t/uint4 readers. Conditions for all rates here: native O3 HIP,
gfx90a, HIP device 0 / PCI 0000:31:00.0, 104 CUs, CPU+memory NUMA node 0,
no model/context, automatic clocks. Sampled active clocks stayed at
1700/1600 MHz core/memory. Useful input-byte GB/s, not DRAM counter traffic.
One shared source is built with `RESEARCH_SCHED_BARRIERS=0` (default) or 1
(matched control); the switch is compile-time, not a runtime kernel branch.

## Removing the compiler scheduling barriers

Each build runs all A/B/C × chunks=1/2/4/8 × grids=104/832 × sizes=1/2/4 GiB,
jointly shuffled in nine rounds of twelve reads, with short/tail controls
and a full checksum after every batch. One process per variant in this
initial experiment; historical repeats are separate, not pooled.

Normal-O3 1 GiB, 104×256 medians:

| Reader / chunks | Barriers on | Barriers off |
|---|---:|---:|
| A / 1 | 956.6 | 1025.1 |
| A / 4 | 1443.7 | 1431.8 |
| A / 8 | 1353.3 | 1022.2 |
| B / 4 | 1428.7 | 1438.6 |
| B / 8 | 1431.6 | 1428.1 |
| C / 4 | 1436.7 | 1023.3 |
| C / 8 | 1353.7 | 1021.5 |

Removing barriers is not a uniform improvement. The whole-body ISA confirms:

- A/4 still issues four 16 B loads before the first consumption wait.
- A/8 changes from eight loads then waits 7..0 to repeated
  **load -> vmcnt(0) -> consume**, despite unchanged 16 B load widths.
- Normal C/4 and C/8 similarly serialize their widened 16 B loads.
- B/4 and B/8 retain sixteen/thirty-two scalar loads before graduated waits.

The data dependency orders each use after its own load; it does not require
the compiler to issue the other independent loads before that use. The
barriers affected that choice of instruction schedule, not GPU memory
completion or thread synchronization.

The vectorization-controlled build also changes when barriers are removed:
C's loads merge to 16 B instructions even with the previous vectorization
controls. It reaches 1355 at chunks=2, but then serializes at chunks=4/8,
giving ~1012/1023. This must not be compared as if it were the old explicitly
scalar C (~675). Removing a scheduling constraint can change load merging
as well as issue/use ordering.

`check-loads.py` now records the complete outer tile-loop load and wait
sequence, separately from the first-load-to-first-wait window. In barrier
controls it still enforces the previous shapes; in the no-barrier experiment
it observes merging and serialization rather than rejecting the very
changes under test. Cache-policy and scratch checks remain strict.

## Winner configuration at 104, 208 and 416 blocks

User-directed narrowed test: **A/B only, four chunks, 256 threads/CTA,
normal O3, scheduling barriers off**. C is omitted because its optimized
representation overlaps A's purpose. Grids are exactly 104/208/416 (not 432).
Each run jointly shuffles reader, grid and 1/2/4 GiB size in nine rounds
of twelve reads; two fresh independent process runs.

1 GiB medians:

| Blocks | A run 1 / run 2 | B run 1 / run 2 |
|---|---:|---:|
| **104** | **1436.4 / 1433.2** | **1438.0 / 1449.2** |
| 208 | 1355.0 / 1354.1 | 1420.0 / 1413.6 |
| 416 | 1308.5 / 1308.6 | 1370.8 / 1371.7 |

4 GiB medians:

| Blocks | A run 1 / run 2 | B run 1 / run 2 |
|---|---:|---:|
| **104** | **1399.6 / 1399.9** | **1401.2 / 1402.2** |
| 208 | 1337.9 / 1337.7 | 1401.2 / 1401.7 |
| 416 | 1301.6 / 1300.2 | 1359.4 / 1359.3 |

**More blocks do not improve these already well-fed readers.** B is more
robust than A at 208; neither benefits at 416. Keep 104×256/chunks=4 as the
winner for this experiment, not a universal geometry policy.

Changing grid size also changes the address schedule: waves=4×blocks,
so successive tiles in one wave's batch are 416/832/1664 KiB apart. These
runs therefore vary both launched wave count and concurrent address
spacing. Do not claim that CU contention alone uniquely explains the loss.
We did not gather working hardware counters to isolate that mechanism.

## Reproduce and artifacts

```sh
research/vram-coalescing/build.sh
# Final argument: off (default) or on, selecting a matched binary.
research/vram-coalescing/run.sh temp/vram-coalescing/test-off default sweep 9 off
research/vram-coalescing/run.sh temp/vram-coalescing/test-on default sweep 9 on
research/vram-coalescing/run.sh temp/vram-coalescing/test-blocks default blocks 9 off
```

`build.sh` produces normal/controlled barrier-free binaries and separate
`*-barriers` controls. Packing also follows the header's barrier-free default;
historical packing data remains labeled and is not retroactively reinterpreted.
The `blocks` phase selects only the two winners and the requested three grids.

Summaries and ISA reports are retained in `results/{no-sched-default,
with-sched-default,no-sched-controlled,with-sched-controlled,blocks-a,
blocks-b}*.csv`. Full sources/binaries/hashes, raw rounds, resource queries,
ISA, device metadata and telemetry are in the corresponding
`temp/vram-coalescing/` directories. No production code or system changes.
