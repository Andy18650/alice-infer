# Bandwidth versus independent VMEM window

User-directed follow-up to the unloaded latency baseline and the
bandwidth-times-latency discussion. Standalone gfx90a research: no production,
package, profiler-source, clock/power-policy or system changes.

## Answer so far

**The unloaded bandwidth-latency product is a useful starting point, not a
sufficient nominal software-window target.** In this tested burst/drain reader,
a nominal 512 KiB window reaches at most ~1.165 TB/s. About 832 KiB nominal
gets close to the observed plateau; 1664 KiB reaches ~1.47-1.49 TB/s.
Native counter captures show the mean VMEM instruction lifetime increasing
as the offered window grows. Thus using the unloaded ~0.34-0.40 us as a fixed
latency is not justified near saturation.

These are **shader VMEM windows**, not measurements of physical bytes in HBM
controller queues. No universal minimum window for every kernel is claimed.

## Controlled reader and measurement

- Native O3 HIP/C++20, one gfx90a GCD: HIP 0, PCI **0000:31:00.0**, 104 CUs.
- CPU+memory NUMA node 0, no model/attention context, automatic clocks.
- **1040 MiB = 1,090,519,040 B** immutable uint32 input, generated outside timing.
  This size permits identical, tail-free work for every tested geometry and
  is much larger than the 8 MiB L2.
- 256 threads/CTA (four wave64s), CTA counts **4/8/16/32/64/104/208/416**.
- A wave instruction loads either **256 B** (one dword/lane) or **1024 B**
  (four dwords/lane). Independent burst depths **1/2/4/8/16**.
- At fixed width/grid, increasing depth changes only how the same per-thread
  address sequence is grouped. The address union, input allocation, checksum
  work and total payload are unchanged. Changing grid/width also changes
  per-wave assignment and issue granularity: those are not pure depth controls.
- Each burst issues exactly D independent global loads, with address-only
  additions between them, then **one vmcnt(0)** before consuming the values.
  There are no per-load timestamps in the throughput kernel. This is a
  deliberately controlled **burst/drain** pipeline, not a continuous ring.
- A separate reader warms the GPU for two seconds; each case has three warm
  launches. Two fresh process runs, differently shuffled case orders, nine
  intervals/case, **four complete reads per interval**. Integer per-thread
  checksums pass an independent closed-form host oracle for every case.
- **80 configurations x 18 samples = 1440 unprofiled records**. A separate
  52 MiB smoke also checks all 80 configurations without replacing main data.

All reported throughput is decimal GB/s or TB/s. Windows use binary KiB/MiB.
The requested-byte numerator omits the small final checksum stores.

## Unprofiled throughput results

1024 B/wave instructions, fixed **104 CTAs /416 waves**:

| Depth | Nominal load window | Process A median GB/s | Process B median GB/s |
|---|---:|---:|---:|
| 1 | 416 KiB | 1017.8 | 1026.5 |
| 2 | 832 KiB | 1414.9 | 1401.9 |
| 4 | 1664 KiB | **1471.8** | **1485.6** |
| 8 | 3328 KiB | 1234.1 | 1361.5 |
| 16 | 6656 KiB | 1353.0 | 1348.7 |

Depth 8 is visibly process-sensitive; its pooled median is not promoted as
one stable physical rate. Both runs are slower than depth 4. The cause of the
between-process spread is not isolated (allocation/history/clock/ordering effects
are not independently controlled). Do not infer a channel map from it.

The best pooled median for 256 B instructions is **1475.6 GB/s** at grid208,
depth16 (process medians 1476.5/1473.1). Both instruction widths can therefore
reach this family's plateau, but need different instruction/wave provisioning.
This is not a reproduction of the historical byte-identical reader's 1530 GB/s;
the code, buffer size and access grouping differ.

### Same nominal window does not imply the same throughput

All rows below have **512 KiB nominal capacity**, fixed 1024 B wave instructions:

| CTA count | Depth | Pooled median GB/s |
|---:|---:|---:|
| 64 | 2 | **1164.9** |
| 32 | 4 | 1038.1 |
| 16 | 8 | 728.2 |
| 8 | 16 | 374.1 |

The narrower-instruction 512 KiB cases reach 930.8 (grid64/depth8) and 763.4
(grid32/depth16). Across these six cases, the best is only ~79% of the best
observed family rate. Distribution across waves/CUs, issue granularity,
address/request ordering and burst drainage all matter; capacity alone does
not establish that the hardware can use it effectively. This does not isolate
which physical memory resource limits any one row.

All tested CTAs can fit the queried resource-capacity upper bound. This is not
a timestamp proof that all CTAs start simultaneously. Registers range up to
70, and every entry has **zero private bytes and zero LDS**: no hidden spill
explains the depth-8/16 behavior.

## Native latency/occupancy counters

The native SDK's built-in **VmemLatency** metric works. The earlier Compute
3.3.0 analysis table was blank; that is not a hardware lack of this metric.
The working native expression is:

```
VmemLatency = reduce(accumulate(SQ_INST_LEVEL_VMEM, HIGH_RES),sum)
              / reduce(SQ_INSTS_VMEM,sum)
```

An additional dependent-probe capture validates 63 dispatches against the
expected wave/VMEM counts and cache/large-footprint regimes. For example,
default wave-dword lifetime is ~103 cycles at 4 KiB, ~244 at 64 KiB, ~610 at
64 MiB and ~695 at 256 MiB. These cover the entire kernel, **including the
cold full-cycle warmup and final stores**, so intermediate fitting-L2 cases
must not be compared blindly with only the timed warm subinterval.
`native-vmem-check.csv` preserves all modes, including store-only controls.

Three independent **single-pass** captures of all 80 window cases collect
latency, VMEM instruction count, waves, active cycles and external/DRAM read
requests together: **240 counter records**. Per-kernel iterations
[4,8,...,32] select each case's fourth launch after its three warm launches.
Raw traces verify every selection. Initial reference warmup has a separate
kernel name and is excluded.

For 1024 B wave instructions at 104 CTAs, counter-pass medians are:

| Depth | Kernel requested GB/s | VMEM lifetime, estimated ns | Mean VMEM payload-equivalent window |
|---|---:|---:|---:|
| 1 | ~1039 | 361 | 367 KiB |
| 2 | ~1449 | **511** | **725 KiB** |
| 4 | ~1514 | **968** | **1436 KiB** |
| 8 | ~1306 | 2174 | 2773 KiB |
| 16 | ~1384 | 2403 | 3249 KiB |

Profiled kernel timestamps and unprofiled HIP-event intervals are different
measurements under different instrumentation. Do not substitute the ~1514
profiled value as a new unprofiled record or mix those columns into an exact
bandwidth-latency equality. Loaded active-cycle/time ratios are approximately
1.63-1.66 GHz, measured in the same dispatch, not assumed nominal 1700 MHz.

### What the inferred window means

Let A be the accumulated outstanding-VMEM-instruction count integrated over
core cycles, N the VMEM instruction count, and T active core cycles:

```
mean instruction lifetime = A / N
mean outstanding instructions = A / T
                              = VmemLatency * N / GRBM_GUI_ACTIVE
payload-equivalent window = mean outstanding instructions * bytes_per_wave_load
```

This measures an active-cycle-weighted **shader instruction population**.
It is not register-buffer allocation or a count of bytes physically waiting
inside HBM. In particular:

- Each coalesced wave instruction produces several lower-level transactions.
  Some fragments may already have returned while the instruction remains
  outstanding. Multiplying by its full payload is a **byte-equivalent**, not
  a precise count of unreturned fragments.
- The lifetime includes shader/cache/return-path queuing; it is not pure DRAM
  service latency or a memory-controller queue measurement.
- The final checksum stores are included in the metric. Their instruction
  count is at most ~0.16% here, but their width/lifetime differs from loads.
  Treat byte conversion as a load-dominated estimate, not exact load-only MLP.
- Ns conversion uses dispatch-average active cycles / elapsed time. Clocks
  are not locked; this is not a per-request steady-counter timestamp.
- The bandwidth-latency product and inferred population share the same
  accumulator/count/time quantities. Their agreement is an **identity**, not
  an independent experimental proof of Little's law. The informative results
  are achieved bandwidth, measured lifetime growth and dependence on request
  distribution.

For example, ~1.45 TB/s x ~0.51 us is ~0.74 MB (~723 KiB), consistent with
that row's instruction-population estimate and larger than an unloaded-latency
budget. Near full throughput, increasing the nominal window further mainly
inflates observed VMEM lifetime; it does not guarantee higher throughput.

## Traffic and code-generation checks

For every counter record:

```
SQ_WAVES = grid * 4
SQ_INSTS_VMEM = input_bytes / bytes_per_wave_load + SQ_WAVES
               independent load instructions       checksum stores
```

All **240** records pass these exact counts. External read payload is calculated
from 64 B requests minus the 32 B-request correction; 32 B counts are zero here.
It matches the full 1040 MiB input within one part per million, with >99.9% of
requests tagged DRAM. Thus cache hits are not inflating the throughput numerator
by re-reading a small resident buffer. This is interface payload, **not physical
wire traffic including ECC/protocol bytes**.

`check-window-codegen.py` verifies all ten actual kernel entries: D load sites
with the intended dword/dwordx4 width, address-only additions until the final
vmcnt(0), no timing instructions, and zero private/LDS storage. The generator
and checked-in private header are byte-compared at build time. Per-thread
checksums alone are not a cryptographic coverage proof; the exact VMEM and
external payload counts are independent additional checks.

Existing mixed-version profiler limitations still apply: resource metadata
from the profiler is not substituted for HIP/ELF, and no unsupported SPM or
PC-sampling claim is made. No profiler-source modification was needed.

## Reproduce / retained artifacts

```sh
bash research/memory-latency/build-window.sh temp/memory-window/new-build
bash research/memory-latency/run-window.sh temp/memory-window/new-a \
  temp/memory-window/new-build/window --mib 1040 --rounds 9 --repeats 4 --seed 12345
bash research/memory-latency/run-window.sh temp/memory-window/new-b \
  temp/memory-window/new-build/window --mib 1040 --rounds 9 --repeats 4 --seed 98765
bash research/memory-latency/profile-window.sh temp/memory-window/new-profile \
  temp/memory-window/new-build/window
python3 research/memory-latency/check-window-codegen.py temp/memory-window/new-build \
  --out temp/memory-window/new-codegen.json
python3 research/memory-latency/analyze-window.py \
  --runs temp/memory-window/new-a temp/memory-window/new-b \
  --profiles temp/memory-window/new-profile --out temp/memory-window/new-summary
```

The profile helper configures the existing local Python/SDK/AQL components;
paths are machine-specific and match the earlier profiler-usability check.
Run/profile helpers verify the build's executable/source/header hashes and
refuse nonempty output directories. Analysis rejects mixed executables,
ambiguous captures, wrong work counts and populations beyond their nominal
bound. `--mib` must be a multiple of 52 for the uniform tail-free grid.

Measured executable SHA256:
`d92363afb3faf2618bb6b805da960b31708b325589a9980c07e4400d3d162e21`.
Curated `results/window-*` CSV/JSON files retain all throughput/counter samples,
medians, source/header/executable hashes and commands; native metric validation
is in `results/native-vmem-check.csv`. Full exact binaries, generated header,
code object/ISA, smoke, raw profiles, commands and logs remain under durable
repo-local `temp/memory-window/` (build, a/b, profile-a/b/c, latency-counter-check-v2).
The native-metric validation repeats the baseline cache-profile command using
`window-counters.yaml` instead of `cache-counters.yaml`; baseline `analyze.py`
with `--counter-log` / `--counter-csv` reproduces the joined 63-row table.
No benchmark source was changed between unprofiled and profiled captures.

## Boundary / next control

This establishes a bandwidth/window curve for a long, contiguous burst reader,
**not a universal hardware window minimum**. A steady ring may have a different
knee. It is also not the actual column-major GEMV stride, FP32 math, SMEM-x
traffic, 32 MiB duration or combine cost. Next useful control: collect these
native lifetime/population counters on that actual GEMV producer before
blindly increasing its prefetch depth. Nothing is integrated into production.
