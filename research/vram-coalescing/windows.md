# Address windows, sustained warmup and clock reporting

User request: compare four 1 GiB windows in the same 4 GiB allocation with
the full-buffer read; warm up long enough to look for hidden clock changes,
including fabric/SoC rather than just graphics and memory DPM states.

## Protocol

Six cases, all 104×256, normal O3 HIP/C++20, phase-only scheduling barriers:
A/four chunks, B/four chunks and B/sixteen chunks, each grid-spaced and
adjacent. Same HIP GPU 0 / PCI 0000:31:00.0, CPU+memory NUMA node 0, no
model/context or clock overrides. No system settings are changed.

Each process allocates and initializes one 4 GiB buffer. It measures:

- `window`: 1 GiB at offsets 0, 1, 2 and 3 GiB;
- `full`: one 4 GiB kernel;
- `split4`: all four windows in succession, four 1 GiB kernels per iteration.

All are twelve timed iterations per batch. Thus split4 has 48 kernel
launches, not twelve; useful bytes per iteration are still 4 GiB. Each
window has a separate checksum output region, validated outside timing.
Inputs and output slices are range checked. Offset-dependent small-input
checks are essential: the affine sum for full GiB windows can coincide
modulo 2^32, so large-buffer checksums alone do not verify correct offsets.

First stage uses the usual 250-read brief warmup (~0.19 s). It is **not a
cold-start experiment**: allocation, initialization and validation precede
it. Second stage uses 30 s of sustained full-buffer reads, cycling the six
cases. Nine jointly shuffled rounds follow each stage. Stages are always
brief then sustained, not randomized or separated by a controlled cooldown.

Two independent rich-telemetry runs (`windows-a/b`) are followed by a basic
telemetry control without rich SMI polling (`windows-basic`). A fourth run
(`windows-curve`) additionally records every sustained-warmup batch's rate.
Each run retains 72 workload/stage groups with nine measurements apiece.
All offset, partial-record, split-output and timed checksums pass.

## Windows do not reveal a consistently bad region

A/adjacent/four chunks, useful-byte median GB/s, rich runs 1 / 2:

| Workload | Brief warmup | After sustained warmup |
|---|---:|---:|
| 1 GiB, offset 0 | 1456.9 / 1464.8 | 1442.1 / 1469.4 |
| 1 GiB, offset 1 GiB | 1480.0 / 1442.1 | 1449.5 / 1467.9 |
| 1 GiB, offset 2 GiB | 1450.6 / 1465.1 | 1443.2 / 1448.5 |
| 1 GiB, offset 3 GiB | 1463.8 / 1459.4 | 1462.1 / 1475.1 |
| One full 4 GiB kernel | 1409.8 / 1408.9 | 1409.6 / 1409.1 |
| Four 1 GiB kernels, 4 GiB total | 1423.3 / 1430.4 | 1431.7 / 1429.2 |

All windows are faster individually than the full-buffer read, and their
rank varies between processes/stages. A permanently slow quarter of the
allocation is not a sufficient explanation. Warmup does not close the gap.
Split4 recovers some bandwidth, but not all of the isolated-window rate.

The basic-polling control reproduces this: sustained individual A windows
1456–1472, full 1410.9, split4 1423.1. Rich SMI polling is not creating the
main gap, though this is not proof of exactly zero observer overhead.

B/grid-spaced/sixteen chunks retains the opposite size effect: sustained
full 1388.8/1385.9, split4 1334.6/1335.9, individual windows ~1332–1341.
The size effect is therefore not a universal long-kernel throttle penalty.
Other reader/layout results are retained in the CSVs.

## No progressive bandwidth loss during the 30 s warmup

`windows-curve` records each twelve-read batch while alternating the six
full-buffer cases. Early (0–5 s) versus late (25–31 s) medians:

| Case | Early GB/s | Late GB/s |
|---|---:|---:|
| A/grid/four chunks | 1403.2 | 1404.8 |
| A/adjacent/four chunks | 1420.5 | 1419.3 |
| B/grid/four chunks | 1341.1 | 1342.7 |
| B/adjacent/four chunks | 1393.4 | 1396.2 |
| B/grid/sixteen chunks | 1387.8 | 1387.1 |
| B/adjacent/sixteen chunks | 1399.2 | 1399.3 |

There is no substantial declining trend as memory temperature rises into
the low 70s C. This rules against a simple warmup-triggered sustained loss
in these exposed conditions, not throttling after arbitrarily longer runs.

## Clocks: distinguish current states from reported averages

A separate native C++ monitor uses the installed **ROCm SMI library and its
official public struct**, not a guessed binary gpu_metrics layout. It
initializes before the GPU run, selects the GPU by its published PCI ID,
then samples at 100 ms. Firmware metrics version is 1.3; all metric queries
succeed. Read-only calls collect current/average clocks, throttle flags,
power/temperatures, firmware timestamps and query latency. The original
500 ms hwmon monitor remains too. Status markers and CPU timestamps permit
stage/workload association; sampled firmware averages are not exact
per-kernel instantaneous frequency traces.

During active work:

- Current GFX **1700 MHz**, reported average usually **~1593–1620 MHz**.
- Current SoC **1090 MHz**, reported average **999 MHz** throughout.
- Current memory **1600 MHz**, reported average **1598–1599 MHz**.
- Both firmware throttle masks remain **zero**.
- FCLK's DPM query succeeds but reports **0 Hz in the deep-sleep entry**, even
  during sustained local reads. That is not interpretable here as the active
  HBM-path fabric clock. It may be an inactive domain/reporting limitation;
  we cannot use it to exclude changes in an unexposed fabric domain.

The current/average discrepancy is real **in the reported fields**; do not
silently equate the two or claim independent physical-clock verification.
Reported averages do not show a warmup- or size-correlated drop: for example,
run A GFX averages have medians ~1604–1606 MHz across brief/sustained
window/full/split phases, with SoC always 999 MHz. Firmware averages were
not captured before the earlier power cycle, so this cannot settle the
historical 1530-versus-1400 shift or prove the average/current discrepancy
is new. No clock-locking or privileged experiment was attempted.

## What this does and does not isolate

The six measured read kernels have identical full instruction sequences
and resources to the preceding order-matrix experiment. Whole code objects
for all four new processes are mutually byte-identical; the old entire ELF
hash differs, so the claim of equality to the old run is specifically the
six reader instruction sequences, not the complete old ELF.

There is no evidence of a warmup/size-linked downclock in exposed clocks,
and no declining warmup bandwidth. Secret fabric downclocking remains
unverified rather than conclusively disproved. More plausible remaining
categories include translation/cache working-set effects and sustained
traversal/request-phase behavior.

Window and split4 controls are not pure duration controls. Restarting a
kernel resets traversal and tile-to-CTA assignment at each window boundary;
1 GiB is not generally an integer number of full 104×Chunks tile batches.
Split4 also adds launches, reductions, tails and changes inter-window reuse.
This explains why the control is informative but cannot uniquely identify
TLB reach, memory-controller phase/contention, or any other one mechanism.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-windows \
  default windows 9 phase rich
research/vram-coalescing/run.sh temp/vram-coalescing/new-windows-basic \
  default windows 9 phase basic
```

Dependencies are already-installed system rocm_smi headers/library;
`clock-monitor.cpp` is compiled by the system C++20 compiler. No installs,
vendored libraries or driver configuration changes.

Tracked: `results/windows-{a,b,basic,curve}.csv`, warmup duration reports,
`windows-clocks.csv`, `windows-warm-bandwidth.csv`, `windows-warm-trend.csv`
and resources. Full source/binary/inspector snapshots, raw rounds,
`clocks.csv`, firmware sample timestamps and telemetry are under matching
`temp/vram-coalescing/` folders. The warm-curve run includes its batch timeline.
The later [four-body control](body-control.md) measures the intermediate
single-kernel/four-body path against these two bulk paths, with explicit
single-operation and twelve-operation timings and separate-output validation.
Baseline and production code remain unchanged.
