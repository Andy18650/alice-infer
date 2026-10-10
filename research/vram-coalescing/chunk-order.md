# Flip chunk ordering: adjacent tiles per block

User proposal: each block stages consecutive tiles, instead of tiles one
full grid apart. **The change matters, but is not uniformly better.**
A/four chunks benefits at 64 blocks, and B/sixteen chunks benefits strongly
at 32 blocks. Larger A batches generally become slower.

## Mapping

Let G be the grid size, C the chunk count, b the block number, and r the
outer iteration. Each tile is `4*blockDim.x` dwords (4 KiB at 256 threads).

```text
Grid-spaced tile: r*C*G + chunk*G + b
Adjacent tile:    r*C*G + b*C + chunk
```

Both traverse all tiles once. The next iteration advances C*G tiles in
both cases. Adjacent means a run of C tiles **per iteration**, not one
single contiguous whole-buffer partition per block. A/B ownership within
each tile stays unchanged. Tail masking covers an incomplete final group.

`read_body<Reader,Chunks,Adjacent>` has a compile-time layout flag, default
false. Ordinary `read_tiles` retains the grid-spaced baseline; the new
`read_adjacent_tiles` entry selects true. No runtime layout branch or
production dispatch change.

## Matched experiment

A/B, 256 threads, grids 104/80 at four chunks and 64/32 at 4/8/16 chunks.
For every configuration, both layouts are interleaved in the same process
and allocation. Native O3 HIP/C++20, gfx90a, HIP device 0 / PCI 0000:31:00.0,
CPU+memory NUMA node 0, no model/context or clock overrides. Each barrier
policy has two independent runs, nine shuffled rounds × twelve reads,
1/2/4 GiB. Sampled active clocks stay at 1700/1600 MHz core/memory.

Phase-only constraints (sched=1, per-load=0), 1 GiB useful-byte median GB/s,
process 1 / process 2:

| Reader | Blocks | Chunks | Grid-spaced | Adjacent |
|---|---:|---:|---:|---:|
| A | 104 | 4 | 1425.7 / 1427.2 | 1453.0 / 1454.8 |
| B | 104 | 4 | 1379.3 / 1367.7 | 1433.7 / 1446.5 |
| A | 64 | 4 | 1339.1 / 1331.0 | **1419.8 / 1423.4** |
| B | 64 | 4 | 1216.1 / 1219.4 | **1295.3 / 1295.6** |
| A | 64 | 8 | 1417.0 / 1424.6 | **1358.9 / 1360.7** |
| B | 64 | 8 | 1278.1 / 1286.7 | 1277.9 / 1283.2 |
| A | 32 | 16 | 1360.1 / 1371.6 | **1271.2 / 1247.1** |
| B | 32 | 16 | 713.4 / 714.0 | **1048.9 / 1062.8** |

At 4 GiB the same key effects persist: A/64/chunks4 is 1338.3/1338.5 →
1407.4/1396.5; A/64/chunks8 is 1385.8/1390.7 → 1355.1/1356.5;
B/32/chunks16 is 709.4/708.4 → 1027.3/1027.7. Adjacent A/80/chunks4
reaches 1454.7/1447.8 at 4 GiB. B does not reach the full plateau at 32.

The per-load-plus-phase policy independently retains the broad result:
A/64/chunks4 1338.8/1338.1 → 1432.4/1435.5, A/64/chunks8
1432.8/1433.3 → 1366.4/1361.4, B/32/chunks16 686.2/686.6 →
1057.2/1050.5. All cases, not just these selected examples, are in the CSVs.

## Assembly and resource changes

Strict ISA checks verify complete batches before the first wait in both
layouts: A 4/8/16 wide 16-byte loads, B 16/32/64 scalar dword loads. No
load→wait-zero→consume serialization, cache-policy changes or scratch-memory
traffic explains the differences.

The compiler generates substantially different address/register code.
For phase-only controls, inspected segments from first load to backward
branch contain:

| Reader/chunks | Grid-spaced lines | Adjacent lines |
|---|---:|---:|
| A/4 | 35 | 32 |
| A/8 | 79 | 60 |
| A/16 | 151 | 124 |
| B/4 | 130 | 70 |
| B/8 | 258 | 150 |
| B/16 | 746 | 309 |

B/16's grid-spaced segment contains 121 VGPR-lane reads and 109 writes
for SGPR spills; adjacent has **none of these hot-loop lane operations**.
It trades for greater vector-register allocation: reported registers
137 → 262, CTA/CU capacity 3 → 1. The full adjacent kernel contains six
accumulator-register reads and six writes outside this captured segment,
so **do not call it spill-free** merely because the old SGPR lane moves
or scratch loads are absent. The inspector now records these too.

A also allocates more registers with adjacency (chunks8: 70 → 88;
chunks16: 134 → 168), despite shorter segments. These are resource
capacities, not observed small-grid placement. This experiment changes
address order **and generated code/resources**; B's improvement cannot be
attributed solely to cache locality or a physical HBM channel mechanism.
No hardware-counter attribution is available.

## Validation / reproduce

Initial short/tail and all timed checksums pass in four nine-round runs.
The final source adds an expanded three-round smoke: empty input, 4 B,
64 B, 68 B, 4160 B, 1 MiB+64 B, and four bytes below/above each whole
C*G-tile batch boundary. This covers partial per-thread records and partial
chunk groups. All 32 cases and 1/2/4 GiB timed checks pass. Tail correctness
is tested independently of full-batch instruction shape.

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-order default order 9 phase
research/vram-coalescing/run.sh temp/vram-coalescing/new-order-all default order 9 on
```

Tracked summaries: `results/order-{phase,all}-{a,b}.csv`, plus resource and
ISA reports for each policy. Full source/header/binary snapshots, raw rounds,
telemetry and extracted A/B/layout loops: `temp/vram-coalescing/` folders
with matching names. `order-tail-smoke` validates the final expanded checks.
Supplemental inspector snapshots preserve original reports and record
accumulator-register counts with hashes appended to metadata.
The subsequent [complete order matrix](order-matrix.md) fills in every
A/B × 104/80/64 × 4/8/16 × both-order cell in two new matched runs.
Baseline and production selection remain unchanged.
