# How small can the streaming-reader grid be?

User-requested tests: winning A/B baselines at **80, 64 and 32 blocks**, with
a contemporaneous 104-block control. Keep four chunks, 256 threads,
block/thread indexing, normal O3 HIP/C++20 and scheduling barriers off.
No ring, geometry-dependent tuning or production dispatch changes.

## Result

**80 is the smallest tested grid retaining near-full bandwidth.** The
measurable decline starts at 64; 32 falls substantially, particularly B.
The precise knee is not established: no counts between 64 and 80 were tested.

Median useful input-byte GB/s, two independent process runs:

| Blocks | A, 1 GiB | B, 1 GiB | A, 4 GiB | B, 4 GiB |
|---|---:|---:|---:|---:|
| 104 control | 1427.1 / 1450.0 | 1424.4 / 1413.6 | 1399.8 / 1399.8 | 1381.6 / 1386.0 |
| **80** | **1461.2 / 1455.0** | **1441.7 / 1442.1** | **1419.7 / 1416.4** | **1378.3 / 1372.1** |
| 64 | 1344.8 / 1342.5 | 1317.9 / 1316.7 | 1348.3 / 1348.5 | 1325.9 / 1325.2 |
| 32 | 1001.9 / 1002.2 | 826.6 / 826.3 | 1003.0 / 1002.8 | 829.4 / 829.4 |

2 GiB results lie in the same bands and are retained in the CSVs. 80 does
not universally win: B is slightly lower at 4 GiB than its 104 control,
while A improves modestly and B improves at 1/2 GiB. Near-equal throughput
is the conclusion, not a universally faster grid policy.

## Conditions and limitations

gfx90a, HIP device 0 / PCI 0000:31:00.0, 104 physical CUs, CPU+memory NUMA
node 0, no model/context or clock overrides. Each process jointly shuffles
reader × grid × 1/2/4 GiB over nine rounds, twelve reads per timed batch,
with checksums following each batch. Active telemetry samples 1700 MHz
core / 1600 MHz memory in both runs.

The entire embedded GPU code object has the **same SHA-256** as the prior
promoted-baseline indexing experiment:
`2d7257a277af94dec258a402d92994b8952833cd1ca55c21c0d014165d02b864`.
Only the host phase/candidate list changes; A/B instructions and resource
allocation are unchanged. A/B report 52/54 registers, zero scratch and
8-CTA/CU occupancy capacity, as before. Small/tail and timed-batch
checksums pass; the existing load-width/cache/scratch checks pass.

Reducing grid size changes **both** available request concurrency and the
per-thread chunk spacing. A 256-thread block tile is 4 KiB, so the grid
strides at 104/80/64/32 are 416/320/256/128 KiB. This experiment does not
isolate active-CU count from address phase/contention effects. In particular,
do not claim an intrinsic 80-CU saturation requirement from the CTA count:
small-grid placement was not separately traced here.

A's wide loads hold up better than B's scalar loads at 32 blocks. This is
consistent with their different instruction/request costs, but not a
counter-backed causal attribution. Likewise, fewer launched blocks do not
by themselves imply a fixed proportion of the total HBM bandwidth.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-small-grid \
  default small-grid 9 off
```

Tracked summaries: `results/small-grid-{a,b}.csv` and
`results/small-grid-resources.csv`. Full snapshots, raw rounds, generated
ISA and telemetry: `temp/vram-coalescing/small-grid-{a,b}/`.
The default baseline selection and previous `blocks`/`index` phases are
unchanged. No production engine or system modifications.
