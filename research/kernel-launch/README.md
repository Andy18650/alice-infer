# Single-GCD kernel / CTA launch timing

Question: are all threads of a grid started together, or can a CTA execute
while later CTAs are still being admitted? Also measure increasing numbers
of kernel launches without conflating that with CTA count.

This is a small standalone HIP C++20/O3/gfx90a study on **GCD 0 only**.
No production dependency, peer access, multi-GPU machinery, or atomics.

## Entry marker

The first source statements read `clock64()` and `wall_clock64()`, then
HIP's `__smid()`. There is **no CTA/grid barrier before these reads**.
One lane per wave writes the timestamps, CU identifier and
`cooperative_groups::this_grid().is_valid()` bit. All lanes in a
wave execute SIMT; writing every lane would mostly duplicate observations
and increase instrumentation traffic. Multiple waves of a CTA are retained.

Disassembly confirms `s_memtime` / `s_memrealtime` occur immediately after
four scalar argument/prologue loads, before record-index calculation or
stores. `__smid()` uses the HIP-provided HW_ID extraction. No `s_barrier`,
private scratch or register spills. This is an early executed-instruction
marker, **not a hardware event marking CTA allocation**.

- `clock64`: core-cycle timestamps, saved raw. Core frequency is dynamic;
  do not convert them using the device's advertised maximum clock.
- `wall_clock64`: constant-rate timestamps used for comparisons/conversion.
  This device reports 25,000 kHz (40 ns ticks). All stamps are on one GCD;
  no assumption about cross-device clock offsets.
- Raw CU IDs are HIP's encoded identifiers, not assumed dense 0..103.
- An optional steady-clock spin holds every wave resident for `hold_us`.
  This deliberately changes resource lifetime to expose residency waves.
- Entry stamps are written immediately; end markers are saved after the
  hold. End markers precede final stores/actual retirement and must not be
  described as exact hardware CTA exit times.

## Separate axes and controls

1. **Grid axis:** 1 / 8 / 32 / 104 / resident-limit / 2×resident-limit CTAs,
   256 threads each, one kernel. Compare zero hold with 100 us hold.
2. **Kernel-count axis:** one CTA per kernel, 1 / 8 / 64 / 256 kernels,
   zero hold. Commands are ordered on one nonblocking stream.
3. Each case is launched normally (`eager` / `graph`) and cooperatively
   (`coop` / `coop-graph`), using **the identical marker kernel**. It contains
   no `grid.sync()` or entry barrier. A graph submits the sequence without
   one host API call per kernel; it is a control for submission gaps.
4. Oversubscribed cooperative shapes exercise the actual runtime API, not
   just a host-side guard. Expected failures have NaN timing fields. Unexpected
   graph acceptance is recorded explicitly, never used as a valid timing.

Three warm runs, seven samples per case. GPU event timing disables system
fences; it includes kernel work and host-fed idle gaps, not just entry spread.
Host enqueue time is recorded separately (graph: amortized per node).
No CPU/GPU clock subtraction is attempted. Raw per-wave traces from the
last sample are saved outside timing, and every stamp is checked.

`cta_entry_spread_us`: latest minus earliest CTA entry, where CTA entry is
the earliest observed wave. `wave_entry_spread_us` uses all waves.
`start_gap_us`: median gap between consecutive kernels' earliest wave starts;
zero means no next kernel exists, not zero launch latency.
`cus_seen`: maximum CU coverage per kernel. `early_finish_kernels` counts,
over sampled kernels, cases where some CTA's **last-wave end marker** is
before another CTA's first entry marker.

## Run

```sh
research/kernel-launch/run.sh

# One explicit case:
numactl --cpunodebind=0 --membind=0 \
  temp/kernel-launch/probe 1664 256 1 100 graph 7 temp/kernel-launch/example
```

```text
probe blocks threads launches hold_us eager|graph|coop|coop-graph [samples=7] [out_dir]
probe --sweep [out_dir]
```

Threads must be a multiple of 64, up to 1024. The extended sweep is 60 cases
in one process, with all initialization reused. Occupancy is queried for the
actual marker kernel and block width. Build/runner use no engine or RCCL;
artifacts and CSV/metadata are under `temp/kernel-launch/`.

## Initial ordinary-launch results (before cooperative-context instrumentation)

Native O3, ROCm 7.1, GCD 0 gfx90a, **104 CUs**, CPU+memory NUMA node 0.
No attention/model context. 256 threads/CTA, seven samples. Marker-kernel
occupancy limit: **832 CTAs = 8 per CU** (this is kernel-specific).
Median entry spread, microseconds:

| CTAs | eager, zero hold | graph, zero hold | eager, 100 us hold | graph, 100 us hold |
|---|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.48 | 0.16 | 0.00 | 0.00 |
| 32 | 0.00 | 0.16 | 0.16 | 0.32 |
| 104 | 0.16 | 0.48 | 0.16 | 0.48 |
| 832 | 1.60 | 0.96 | 1.60 | 1.12 |
| 1664 | 2.88 | 2.24 | 102.56 | 101.92 |

Zero is indistinguishable at the steady counter's resolution, not proof of
exact simultaneous start. At 104 CTAs, 104 distinct CUs are observed.
At 832 zero-hold CTAs, an early CTA reaches all its end markers before a
late CTA begins in 5/7 eager and 7/7 graph samples. At 1664 it happens in
all samples. With 100 us hold, 832 CTAs fit concurrently and never show
that ordering; 1664 CTAs do, with a ~100 us residency wave.

### Increasing kernel count

One CTA ×256 threads, zero hold. Values are microseconds:

| kernels | eager GPU/kernel | graph GPU/kernel | eager start gap | graph start gap |
|---|---:|---:|---:|---:|
| 1 | 7.84 | 15.04 | n/a | n/a |
| 8 | 2.78 | 3.06 | 2.24 | 1.44 |
| 64 | 2.83 | 1.55 | 2.24 | 1.28 |
| 256 | 3.25 | 1.51 | 2.24 | 1.28 |

At 256 kernels, host enqueue/kernel is 2.96 us eager versus 0.143 us graph
(amortized). Single-node graph launch is more expensive here; its advantage
is amortization across a sequence. These are **instrumented marker kernels**,
not empty-kernel overhead or a predicted transformer latency.

All 30 cases pass stamp checks; a 208-CTA/64-thread/8-kernel/10-us hold
case also passes per-wave indexing and order checks. Raw timestamps/CU IDs
are retained alongside summary.csv under temp/kernel-launch/results/.

## Cooperative launch comparison and runtime inconsistency

O3, GCD 0, CPU+memory NUMA node 0, no attention/model context, 256 threads,
seven samples. All modes use the same updated marker with a device cooperative
context bit (the earlier ordinary table predates this extra query). Cooperative
modes are confirmed `is_valid()==1`; normal modes are confirmed zero.

Resident grids do not show a generally tighter start interval:

| CTAs / hold | eager spread | graph spread | coop spread | coop-graph spread |
|---|---:|---:|---:|---:|
| 104 / 0 us | 0.16 | 0.48 | 1.28 | 1.28 |
| 832 / 0 us | 1.44 | 1.12 | 1.44 | 1.44 |
| 104 / 100 us | 0.16 | 0.48 | 1.28 | 1.28 |
| 832 / 100 us | 1.60 | 1.12 | 1.44 | 1.60 |

Cooperative launch promises a legal resident grid, not simultaneous wave
execution; it inserts no user entry barrier. It is much more expensive here,
even though the marker contains **no grid synchronization**. For 256 one-CTA,
zero-hold kernels:

| mode | GPU us/kernel | median start gap us |
|---|---:|---:|
| eager | 3.53 | 3.52 |
| graph | 1.83 | 1.60 |
| coop | 19.38 | 19.36 |
| coop-graph | 19.40 | 19.36 |

This isolates launch mode, not the cost of `grid.sync()`. It is consistent
with the P2P study's identical-kernel cooperative control, but does not identify
the exact runtime/device dispatch mechanism without further tracing.

### Oversubscription: direct API rejects; captured replay unexpectedly accepts

The marker's limit remains **832 CTAs at 256 threads**. Direct cooperative
launch of 1664 CTAs returns exactly `hipErrorCooperativeLaunchTooLarge`.
Both zero-hold and 100-us-hold cases produce `EXPECTED_FAIL_launch_...` rows.

**Captured cooperative replay accepts and completes the same 1664-CTA shape.**
The device marker confirms a cooperative context (`is_valid()==1`), so this
is not simply an ordinary launch substituted by the probe. It is recorded as
`UNEXPECTED_ACCEPTED_replay_device_coop1`, with NaN timing fields. No fallback
or relabeling of the result as a legal cooperative benchmark.

The public captured kernel-node attribute getter reports cooperative=0 even
for valid captures whose device context is cooperative=1. Therefore the getter
alone cannot establish the effective mode. The local upstream ROCm 7.1 source
is consistent with two separate fields: `capturehipLaunchCooperativeKernel`
passes an internal `coopKernel_` flag; public `kernelAttr_.cooperative` is
separate; graph parameter validation passes a zero cooperative flag while
command creation retains `coopKernel_`. It also disables graph packet capture
for cooperative nodes. These are source-inspection clues, not a claim that
that checkout exactly matches the installed Canonical runtime binary.

**Do not rely on graph acceptance to validate cooperative residency on this
stack.** Validate occupancy in an application before using `grid.sync()`.
The diagnostic marker has no such barrier, which is why unexpected acceptance
can be observed safely. We do not test oversized grid synchronization or patch
HIP. Ordinary production kernels and graphs are unaffected by this study.

Exit codes: 0 for valid timings / correctly rejected oversize calls; 2 for
safely recorded unexpected cooperative acceptance; 1 for other errors. The
runner preserves output and propagates exit 2 rather than claiming success.
The complete extended sweep reports **56 valid timings, 2 expected direct
failures, 2 unexpected graph acceptances**. Logs/traces:
`temp/kernel-launch/cooperative-results/`. Additional 64/1024-thread checks
pass. Source and binary hashes are recorded by run.sh.

## What this answers—and what it does not

- There is no observable all-grid start barrier. CTAs/waves begin over a
  finite interval, and bodies can finish before later CTAs enter.
- Below available residency, many CUs begin very close together. Above it,
  a deliberate hold produces clear subsequent admissions when slots free.
- This is consistent with **GPU workgroup dispatch and scheduling as resources
  allow**, not CPU per-CTA launch calls. Host eager submission time stays
  roughly 1.1–1.5 us for a single grid from 1 to 1664 CTAs; it does not scale
  like a host setup operation per CTA.
- A kernel launch describes a grid. The driver/runtime and GPU dispatcher
  are different layers; entry timestamps cannot identify exactly which
  hardware setup pipeline allocates registers, LDS, or wave state.
- Wave timestamps include scheduling/prologue/clock-read effects. They do
  not prove that CTA IDs start in numerical order or all waves of a CTA
  begin together. No entry synchronization was inserted to manufacture it.
- The same-stream kernel-count test orders whole kernels; it does not test
  concurrent multi-stream execution, priorities, or cooperative launch.

Production code is unchanged. Counter-visibility research was committed
separately as `a75ef23` before this study.
