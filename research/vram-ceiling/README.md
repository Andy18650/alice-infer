# Local VRAM streaming-read ceiling

Question: is the decode-roofline study's 1290 GB/s a hardware ceiling, or
merely what one read kernel achieved? Start with local reads, not GEMV,
peer access, copy bandwidth, or an engine optimization.

## Result

These are the original campaign's historical results. A [post-power-cycle
replay](replay.md) rebuilds a byte-identical executable but measures the exact
old winner at ~1406–1413 GB/s. The earlier 1530 result is not a reproducible
current baseline; the cause of the shift remains unisolated.

**1290 GB/s is not the local streaming-read ceiling.** Two final independent
process runs confirm >1500 GB/s on a 1 GiB working set and approximately
1460 GB/s on a 4 GiB working set. The best nine-round 1 GiB confirmation
median was **1530.5 GB/s** (+18.6% over the historical 1290). A repeat's best
1 GiB median was 1509.3. Do not advertise the screening maximum as a stable
ceiling, or extrapolate the 1 GiB winner to larger buffers.

Conditions for every number here: standalone native **O3 HIP/C++20**,
MI250X/gfx90a, **HIP logical device 0, PCI 0000:31:00.0, 104 CUs**;
`numactl --cpunodebind=0 --membind=0`; **no model or attention context**.
ROCm/HIP 7.1.52801, Ubuntu clang 21.1.8. No clock or power settings changed.
During both final runs, 0.5 s telemetry samples read sclk=1700 MHz,
mclk=1600 MHz, busy=99–100%; sampled memory temperature rose to 73/75 C.
HIP ordering differs from rocm-smi: this device is rocm-smi GPU **2**, not 0.
Before/after idle clock snapshots alone would misdescribe the timed clocks.

Selected confirmation medians (decimal GB/s; nine batches of 12 reads each):

| Global-grid pattern; load bytes / unroll / threads per CTA / CTAs | Working set | Run 1 | Run 2 |
|---|---|---:|---:|
| 4 / 1 / 256 / 832 (old geometry, new integer checksum) | 1 GiB | 1367.3 | 1374.5 |
| 4 / 1 / 256 / 832 | 2 GiB | 1315.9 | 1324.6 |
| 4 / 1 / 256 / 832 | 4 GiB | 1310.6 | 1300.9 |
| 4 / 16 / 256 / 208 | 1 GiB | 1486.7 | 1508.7 |
| 4 / 16 / 256 / 208 | 2 GiB | 1454.0 | 1450.1 |
| 4 / 16 / 256 / 208 | 4 GiB | 1460.7 | 1451.0 |
| 8 / 1 / 512 / 832 | 1 GiB | 1476.8 | 1466.2 |
| 8 / 1 / 512 / 832 | 2 GiB | 1507.1 | 1480.8 |
| 8 / 1 / 512 / 832 | 4 GiB | **1462.8** | **1460.2** |

The 4 GiB 8-byte-load control is a useful conservative reference: run 1
range 1426.3–1505.1, run 2 range 1419.1–1487.9 GB/s. Its medians are about
13% above 1290. Different geometries top the 1 GiB rankings: run 1's
8/16/256/832 median is 1530.5, range 1453.3–1532.0; run 2's
4/16/64/832 median is 1509.3, range 1437.8–1532.2. Variance is real.

Summary rows for all final confirmation configurations are retained in
`results/run-{1,2}.csv`. Full raw batches, device/compiler/source/binary
hashes, preserved source and executable, telemetry, and management snapshots:
`temp/vram-ceiling/final-{1,2}/` (gitignored). Preliminary runs using
sequential integer input remain under `results/` and `repeat/` in that temp
directory; the table above uses the final dispersed-word initializer only.

## Sweep

`ceiling.hip` is standalone: no engine headers, objects, rocBLAS, or CMake
build dependency. Allocation, initialization, and checksum readback are
outside timing. All buffers are ordinary `hipMalloc` device memory.

576 cases, in a fixed-seed shuffled order:

- load width: 4 / 8 / 16 bytes (`uint32_t`, `uint2`, `uint4`);
- independent accumulator chains / loop unroll: 1 / 4 / 8 / 16;
- threads per CTA: 64 / 128 / 256 / 512;
- grid: observed CU count × 1 / 2 / 4 / 8 / 16 / 32;
- traversal: global grid-stride vs separate contiguous partition per CTA.

Each vector component has an independent integer accumulator. One sum per
native wave is written to a **unique** output slot: no old shared-sink race.
Input word i is `uint32(i)*1664525 + 1013904223`, avoiding all-zero or
identical-row data. A host-computed analytic modulo-2^32 checksum validates
the combined output from every wave. This is a checksum, not an exhaustive
proof against every possible error. All 576 variants additionally pass
64-byte, 4160-byte and 1 MiB+64-byte controls, exercising short loops,
empty partitions and tails.

Preheat: 250 reads of 1 GiB. Screening: three warmups, checksum, three
HIP-event batches of 12 reads, median used to select 12 finalists. All
finalists plus the old-geometry scalar control are remeasured at 1/2/4 GiB
in nine independently shuffled paired rounds per size, three warmups and
12 timed reads per round. Checksum readback follows each confirmation
batch. Rates count **input bytes only**; event time includes the final
wave reduction and sparse checksum stores. They are not counters of
physical HBM transactions. Buffer sizes greatly exceed the device's L2;
there is no explicit cache flush between streaming reads.

The global pattern won screening in both final runs. The best slab cases
were 1405.5 / 1366.5 GB/s and did not enter the top 12. This is not proof
that contiguous per-CTA work is intrinsically inferior: partition alignment
and address scheduling differ as well. Slab sizes are divided evenly in
vector units, not rounded to transaction boundaries.

## Codegen check

Disassembled the **actual final executable's** embedded gfx90a object:

```sh
llvm-objdump --offloading temp/vram-ceiling/final-2/ceiling
llvm-objdump -d --demangle \
  temp/vram-ceiling/final-2/ceiling.0.hipv4-amdgcn-amd-amdhsa--gfx90a \
  > temp/vram-ceiling/final-2/disassembly.txt
```

Global kernels retain their intended `global_load_dword`,
`global_load_dwordx2`, and `global_load_dwordx4` instructions. The unroll-16
versions have 16 loads in the main body plus the scalar-iteration tail;
there are no scratch load/store instructions in these checked kernels.
Wide loads have not been silently scalarized, and checksummed input loads
have not been optimized away. Unroll is a source scheduling knob, not a
claim that all those loads are concurrently outstanding at runtime.

## Occupancy check: do all 832 CTAs fit and overlap?

**Yes, for the 8-byte-load/unroll-16 kernel at 256 threads per CTA.**
Do not infer this from grid size alone. The occupancy query loads the
preserved final-1 executable's actual gfx90a code object using the HIP
module API; it does not substitute a freshly compiled kernel.

Exact `read_stream<2,16,false>`: 59 VGPRs, 53 SGPRs in code-object metadata,
zero LDS and private/scratch allocation. HIP reports 8 active CTAs/CU at
256 threads, 32 wave64 waves/CU and 2048 resident threads/CU. The device's
reported maximum is 2048 threads/CU: this launch can reach 100% resident
wave occupancy. Across 104 CUs, the full-CTA capacity is **832**. At 512
threads/CTA the same kernel has capacity **416**, not 832.

Capacity is not observed occupancy, so `occupancy-timeline.hip` adds
per-wave entry/exit `wall_clock64()` markers and CU IDs to a private twin
of the same traversal/accumulation/unroll/tail/checksum. It adds no hold,
barrier, atomic, cooperative launch or grid synchronization. The twin also
uses 59 VGPRs, zero LDS/scratch, and reports the same 8-CTA capacity.
The runtime supplies the steady clock rate (25 MHz on this device).

Two final process runs, seven checked samples per size/grid; native O3,
HIP logical device 0, CPU+memory NUMA node 0, no model/context, eager:

| Grid, 256 threads/CTA | Peak overlapping waves | Peak CTA intervals | 1 GiB all-wave common interval, median A/B |
|---|---:|---:|---:|
| 416 CTAs | 1664 | 416 | 724.0 / 726.6 us |
| **832 CTAs** | **3328** | **832** | **694.2 / 692.0 us** |
| 1664 CTAs | 3328 | 832 | 0 / 0 us |

At 832 CTAs, all 104 CUs show 8 overlapping CTAs / 32 waves. First-wave
CTA entry spread is about 1.6 us; every wave has entered before any wave's
exit marker. At 4 GiB, the common interval is approximately 2965/2950 us.
The 1664-CTA control instead shows about 364/366 us entry spread at 1 GiB
and 1490/1500 us at 4 GiB, with no interval where the whole grid overlaps.

These are software-marked intervals, not exact allocation/release times or
hardware occupancy counters. The original historical launch is not
retroactively traced. The strongest defensible statement is: **the exact
original kernel has capacity for all 832 CTAs, and a same-occupancy,
timestamped reproduction confirms all 3328 waves overlap**. Resident waves
can be waiting on memory; they are not all issuing instructions in one cycle.

In the oversized 4 GiB control, some CUs briefly show nine overlapping
*CTA intervals* while wave overlap remains bounded by 32. A partially
finished CTA's remaining waves can overlap newly admitted work: counting
CTAs with any wave alive is not the same as counting full CTAs' wave
resources. The first exploratory tracer incorrectly asserted the former
against the full-CTA occupancy limit and stopped; final analysis checks
wave limits and records both metrics. No failed result is used as proof
of a higher occupancy limit.

Exact query and final raw traces/source/binaries/hashes are in
`temp/vram-ceiling/occupancy/final/`; summary copies are retained in
`results/occupancy-{a,b}.csv`. Reproduce against a preserved executable:

```sh
research/vram-ceiling/check-occupancy.sh \\
  temp/vram-ceiling/final-1/ceiling temp/vram-ceiling/occupancy/new-run
```

`occupancy.cpp` is the exact-code-object query; the shell script extracts
and records metadata, builds the query and timestamped twin, and runs two
process repeats. Production code and the original ceiling kernel are
unchanged.

## Interpretation and limits

- The historical 1290 was a probe result, not a demonstrated device limit.
- More independent chains can improve even 4-byte loads; simply maximizing
  load width, block size, or total thread count does not identify a winner.
- The baseline here retains the old geometry but uses integer accumulation
  and a safe wave checksum, **not an exact replay of the old float kernel**.
  Attribution of the historical gap to one knob would require further controls.
- Larger working sets change ranking. In particular 8/16/512/832 drops to
  about 1206 GB/s at 4 GiB in run 2 despite a respectable 1 GiB result.
  This study does not isolate the reason; do not name a channel mapping
  from bandwidth alone.
- This is one physical GCD, read-heavy traffic, one allocator/base address
  per process, sequential addresses and automatic clocks. It does not
  establish all-GCD, mixed-read/write, random-access, or GEMV ceilings.
- Fixed size order, preheating and best-of-screen selection remain sources
  of bias. Confirmation and a second process reduce, but do not eliminate,
  those limitations. No memory-controller counters were collected.

For subsequent local-memory roofline work, **~1.46 TB/s for multi-GiB
streaming reads** is a better provisional reference than 1.29; **~1.53 TB/s
is demonstrated on 1 GiB**, not a universal sustainable ceiling. Recheck
at the relevant kernel's traffic size/pattern. No production change follows
from this result.

## Reproduce

```sh
research/vram-ceiling/run.sh temp/vram-ceiling/new-run 0
python3 research/vram-ceiling/summarize.py temp/vram-ceiling/new-run
```

`build.sh` uses hipcc `-O3 -std=c++20 --offload-arch=gfx90a`; outputs live
under durable repo-local `temp/`, never `/tmp`. The runner captures source,
executable, hashes and read-only telemetry. Python helpers use only the
standard library. A final sweep plus confirmation took about 37 seconds
(excluding compilation and management snapshots); other machine activity
can change that. A checksum failure or HIP error exits nonzero.
