# More loads per wave at smaller grids

User hypothesis: increase the in-flight window to compensate for fewer
launched waves. Sweep A/B at **64/32 blocks × 4/8/16 chunks**, plus
104-block/four-chunk controls, always 256 threads and block/thread indexing.

## Finding

**It compensates well for A at 64 blocks.** Eight chunks recover near-full
read bandwidth. At 32 blocks, sixteen chunks recover a substantial fraction,
but the 4 GiB rates remain below the plateau. B improves at eight chunks,
then gets worse at sixteen; compiler/resource costs confound that case.

Normal O3 HIP/C++20, gfx90a, HIP device 0 / PCI 0000:31:00.0, CPU+memory
NUMA node 0, no model/context or clock overrides. Each policy has two
independent processes, nine shuffled rounds × twelve reads, jointly over
reader/grid/chunks and 1/2/4 GiB. Short/tail and all timed checksums pass.
Active telemetry samples core 1700 MHz / memory 1600 MHz.

## Initial per-load barrier policy

`RESEARCH_SCHED_BARRIERS=1`, `RESEARCH_LOAD_BARRIERS=1`: keep existing
per-load constraints and load/consume phase boundaries. All use AMD compiler
scheduling barriers with mask zero: not synchronization, memory fences or
completion waits.

1 GiB useful-byte median GB/s, process 1 / process 2:

| Blocks | Chunks | A (wide) | B (scalar-striped) |
|---|---|---:|---:|
| 104 control | 4 | 1420.9 / 1421.1 | 1373.0 / 1364.6 |
| 64 | 4 | 1333.7 / 1335.5 | 1219.8 / 1219.7 |
| 64 | 8 | **1438.2 / 1418.8** | 1279.9 / 1278.6 |
| 64 | 16 | 1402.8 / 1411.8 | 1225.8 / 1225.8 |
| 32 | 4 | 964.1 / 963.7 | 660.1 / 660.1 |
| 32 | 8 | 1301.1 / 1295.3 | 834.2 / 839.2 |
| 32 | 16 | **1348.1 / 1383.5** | 687.9 / 686.9 |

At 4 GiB, A/64/chunks8 repeats at **1400.3/1397.4**, versus its 104/chunks4
controls 1400.3/1401.1. A/32/chunks16 reaches 1306.4/1299.6, versus
967.3/966.9 at chunks4. The small-input best-case is not a universal plateau.

## Assembly check of abnormal B performance

All staged loads precede the first consumption wait:

| Chunks | A VMEM instructions | B VMEM instructions |
|---|---:|---:|
| 4 | 4 × 16 B | 16 × 4 B |
| 8 | 8 × 16 B | 32 × 4 B |
| 16 | 16 × 16 B | 64 × 4 B |

No old load → vmcnt(0) → consume serialization is present in the issue
phase. A consumes with descending vmcnt thresholds 3..0, 7..0 or 15..0;
B with 15..0, 31..0 or 62..0. Positive thresholds allow requests to remain
pending; they do not prove the actual instantaneous hardware queue depth.
The batches drain at the end before the next iteration; this is not a ring.

B/chunks16 has **SGPR spills into VGPR lanes**, visible as repeated
`v_writelane_b32` saves and `v_readlane_b32` restores. There is still **zero
scratch-memory traffic**: that alone does not imply spill-free code.
Its captured loop segment contains 126 reads / 115 writes of VGPR lanes,
and 760 instruction lines, versus 261 lines and no such lane operations
for B/chunks8. Segment counts start at the first global load and stop at
the backward branch, excluding pre-load address setup.

Reported registers at chunks4/8/16 are A 44/70/134, B 40/70/137; occupancy
capacity is 8/7/3 CTAs per CU. This is capacity, not measured placement;
small grids are not expected to populate that capacity. No spill/bandwidth
causal isolation was done: scalar address/issue/wait costs also change.

## Less restrictive phase-only barriers

To check overconstraint, retain the load/consume boundaries but remove the
individual reader barriers: sched=1, load=0 (`phase` runner option).
The strict ISA checks still verify the complete batches above.

1 GiB medians, two processes:

| Blocks | Chunks | A | B |
|---|---|---:|---:|
| 104 control | 4 | 1446.5 / 1424.9 | 1387.2 / 1373.4 |
| 64 | 4 | 1334.4 / 1331.2 | 1220.3 / 1218.6 |
| 64 | 8 | **1442.6 / 1437.8** | 1293.2 / 1281.5 |
| 64 | 16 | 1365.6 / 1366.3 | 1236.5 / 1235.7 |
| 32 | 4 | 947.5 / 947.5 | 660.0 / 659.7 |
| 32 | 8 | 1293.5 / 1285.3 | 834.8 / 836.7 |
| 32 | 16 | **1375.6 / 1357.7** | 712.3 / 715.0 |

B/chunks16 still spills SGPRs (121 lane reads / 109 writes in the captured
segment); removing the per-load constraints does not eliminate the problem.
A's segments shorten, but bandwidth does not improve universally. At
4 GiB A/32/chunks16 is 1322.4/1308.6 and A/64/chunks8 1383.5/1386.9.

Even four-chunk B is slower with these constraints than in the prior
barrier-free small-grid study: 32-block B was ~827–829, not ~660. Do not
interpret the constrained sweep as a hardware scalar-read ceiling, or its
16-chunk regression as failure of the concurrency idea alone.

## Interpretation / reproduction

Increasing chunks at fixed grid leaves inter-chunk spacing unchanged
(256 KiB at 64 blocks, 128 KiB at 32), while changing batch span, loop
frequency, register liveness and instruction schedule. This is cleaner
than changing grid size, but still not a single-variable hardware isolation.
Enough per-wave ILP can replace some launch-level parallelism; more is not
monotonically better, and source batching must be verified in ISA.

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-inflight default inflight 9 on
research/vram-coalescing/run.sh temp/vram-coalescing/new-inflight-phase default inflight 9 phase
```

Tracked: `results/inflight-{a,b,phase-a,phase-b}.csv`, policy resource/load
reports. Full binary/source snapshots, telemetry and extracted A/B loop
ISA: `temp/vram-coalescing/inflight-{a,b,phase-a,phase-b}/`.
Supplemental checker snapshots and lane-count reports are saved alongside
the original checks, with their hashes appended to metadata.
The ordinary baseline remains barrier-free; no production dispatch, engine,
package or system changes.
