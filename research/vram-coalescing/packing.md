# Where do the A/B/C waves actually run?

**Observed: blocks spread evenly over CUs, and each 256-thread block places
one wave on each of its CU's four SIMDs.** Larger grids then fill more wave
slots on those same CUs. This was measured, not inferred from grid size.

Conditions: native O3 HIP/C++20/gfx90a, HIP device 0, PCI 0000:31:00.0,
104 CUs, CPU+memory NUMA node 0, no model/context. The actual A/B/C read
body processes 1 GiB, four independent tiles, 256 threads per CTA. Controlled
scalar-width compiler flags match the coalescing study. No clock overrides,
artificial holds, atomics, CTA/grid synchronization or cooperative launch.

## Repeated result

Two independent process runs, seven shuffled samples per reader/grid:
252 checked traces. A, B and C agree in every sample through the 832-CTA
capacity. All four waves of every CTA share a CU and use distinct SIMD IDs
0/1/2/3.

| Launched CTAs | CUs used | Assigned CTAs per used CU | Peak overlapping CTAs/CU | Peak waves/SIMD |
|---|---:|---:|---:|---:|
| 52 | 52 | 1 | 1 | 1 |
| 104 | 104 | 1 | 1 | 1 |
| 208 | 104 | 2 | 2 | 2 |
| 416 | 104 | 4 | 4 | 4 |
| 832 | 104 | 8 | 8 | 8 |
| 1664 | 104 | 15–17, usually 16 | 8 | 8 |

At 104 CTAs all 416 wave intervals overlap: **one CTA/CU, four waves/CU,
one wave/SIMD**. At 832, all 3328 wave intervals overlap: **eight CTAs/CU,
32 waves/CU, eight waves/SIMD**. Hardware wave-slot IDs distinguish these
co-resident waves; no same-CU/SIMD/slot pair overlaps with itself.

At 1664, total assignment can be slightly uneven as later CTAs use CUs
whose earlier work has finished. Peak wave overlap remains 3328, with no
interval where all 6656 waves overlap. It is not two globally synchronized
batches, nor sixteen CTAs resident per CU.

The earlier scalar-C bandwidth loss is therefore **not caused by C being
packed onto fewer CUs or fewer SIMDs than A/B**. Identical packing does not
eliminate contention: the waves still share CU resources, and their memory
instruction patterns can put different pressure on LSU/cache interfaces.

## Measurement

`packing.hip` wraps the shared `read_body<Reader,4>` with one record per wave:

- entry and exit `wall_clock64()` values;
- entry and exit full `HW_REG_HW_ID` values;
- CTA index and logical wave index reconstructed from the record offset.

Hardware fields for gfx90a are documented in the installed HIP header
`/usr/include/hip/amd_detail/amd_device_functions.h` (GCN/CDNA HW_ID table):
wave slot bits 3:0, SIMD bits 5:4, CU-local bits 11:8, shader-array bit 12,
shader-engine bits 15:13. The probe reads register 4 with
`__builtin_amdgcn_s_getreg`; host analysis preserves SE/SH/CU in the CU key
rather than assuming contiguous CU numbering or dropping an array bit.
This decoder is explicitly gfx90a-only, not valid for arbitrary RDNA GPUs.

Disassembly retains **two HW_ID reads and two steady-clock reads** in each
marked kernel. Entry/exit CU/SIMD/slot identity is checked. The runtime
supplies the clock rate (25 MHz here); no guessed core-clock conversion.
All wave-output checksums pass before a trace is accepted.

Resource queries on the unmarked and marked kernels:

| Reader | Unmarked / marked reported registers | Unmarked / marked CTA capacity per CU |
|---|---:|---:|
| A, chunks=4 | 50 / 50 | 8 / 8 |
| B, chunks=4 | 52 / 52 | 8 / 8 |
| C, chunks=4 | 52 / 52 | 8 / 8 |

All have zero LDS and scratch allocation. These equal limits make the
instrumented reproduction representative of the same residency class,
but the recorded timeline is still an instrumented kernel, not a hardware
trace of a historical unmodified invocation.

`packing-summary.py` separates:

- **assignment**: how many CTAs ultimately execute on each CU;
- **overlap**: simultaneous marked CTA/wave intervals per CU and SIMD;
- **slot reuse**: verifies no overlapping intervals for a physical slot;
- the whole-grid common interval and entry spread.

A CTA interval spans its first wave's entry to its last wave's exit. That
can include partially draining CTAs; do not generally equate a count of
such intervals with the full-CTA occupancy calculation. For this campaign,
peak CTA counts happen to stay within eight. Software markers lie inside
wave execution, not at exact hardware resource allocation/release times.
Preemption during a marked interval would complicate a residency inference;
these are isolated single-dispatch tests, with no detected placement changes
or overlapping slot reuse. Resident waves may be waiting on memory; they
are **not all issuing instructions in the same cycle**.

## What this says about the scheduler

ROCm submits the dispatch and queue/resource configuration. GPU workgroup
assignment and wave scheduling determine placement. In this workload the
result is balanced spreading, followed by co-residence as the grid grows.
It is **not a guarantee that the scheduler minimizes memory contention**,
or that other kernels, CU masks, priorities, resource footprints, concurrent
work or short-lived CTAs will produce the same distribution.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/packing-run.sh temp/vram-coalescing/new-packing 7
```

Raw per-wave traces, resources, manifest, source/binary snapshots, hashes,
ISA and telemetry: `temp/vram-coalescing/packing-{a,b}/`. First three-sample
exploration: `packing-first/`. Curated summaries and resource table:
`results/packing-{a,b}.csv`, `packing-resources.csv`.

Readers/traversal were extracted into private `readers.hpp` so the bandwidth
and packing probes share exactly one body. Normalized unmarked instruction
sequences for all twelve A/B/C × chunks=1/2/4/8 controlled specializations
match the pre-refactor executable; the bandwidth runner's load-width checks
and checksum smoke test also pass. Historical snapshots remain intact.
Local `.clangd` flags supply HIP context for this research directory, which
is outside the engine's compilation database. No production engine or
system changes.

### Raw-integer revalidation

The shared body now uses raw `uint32_t` input, built-in `uint4` loads and
accumulation, with no float conversions or custom vector aliases. The
original two-run tables above remain historical evidence. A new seven-sample
process (`temp/vram-coalescing/uint-packing/`) repeats all three readers and
six grid sizes: CU/SIMD spreading and capacities are unchanged, with A at
50 registers and B/C at 52, both marked and unmarked. New summary/resource
copies are `results/uint-packing*.csv`.
