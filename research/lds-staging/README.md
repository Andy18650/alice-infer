# Pure global-to-LDS staging, no periodic waits

User-directed standalone gfx90a research, separate from fp16-gemm64. First
measure pure loading, without MFMA, LDS operand reads, consumers, barriers
or explicit waits in the issue loop. Let hardware issue backpressure act.
This is not a direct measurement of maximum outstanding request count.

Next user-directed test: [one load +VM0, register-retained latency samples](LATENCY.md).
It has a separate binary/source and reports all individual intervals with
min/max/median/mean/spread, timing overhead included. This no-wait baseline
remains unchanged.

The [disjoint-source CTA sweep](GRID_LATENCY.md) increases1–256 CTAs with
1 MiB/CTA, beyond the reported8 MiB L2, preserving the per-load timing asm.
It also identifies the slower first source traversal versus faster revisits
in the earlier single-CTA aggregate; prior warmup did not guarantee uniformly
hot measured source accesses. Both pilot and corrected64-batch data remain.

[Per-load VM watermark sweep](WINDOWS.md) returns to long streaming bandwidth:
1/8 CTAs, thresholds0..32 plus larger/disabled/no-wait controls.1 CTA reaches
the floor at vmcnt16 (peak17 loads/wave);8 CTAs approach a noisier plateau at
vmcnt19–20 (peak20–21). Full native-controlled curves and provenance retained;
plateau is sufficiency evidence, not an exact physical queue-depth proof.

The next [DWORDx4 ->VGPR ring study](VGPR_RING.md) uses one CTA with1/2/4
waves,32 register slots and no periodic waits. All three sustain~63.9
requested B/core-cycle with a fixed1 MiB source; per-step payload and period
scale together, so more waves do not raise aggregate rate in this setup.

[Coarse VGPR CTA knees](GRID_VGPR.md) then sweeps1–208 CTAs at1/2/4 waves:
early bend around4–8, broad best region32–64, regressions at104–208
(especially four-wave CTAs). No periodic waits added or follow-on tuning;
source footprint/concurrency remain coupled.

[64-CTA8x8 outer-product sharing](BROADCAST_VGPR.md) replaces disjoint
1MiB/CTA regions with shared512KiB A-row/B-column panels. Same kernel/loop,
8MiB distinct active source versus64MiB, GPU source bases verified. Requested
bandwidth improves~2.2x; not a physical HBM rate or measured GEMM speedup.

## Fixed experiment

One CTA/256 threads/four waves, HIP0/BDF31:00.0, native HIP O3/C++20,
CPU+memory NUMA0, no model/context/clock-policy overrides.

- Source allocation **1 MiB**, four disjoint256 KiB wave ranges. Same
  addresses in both modes; source aliasing is not another axis.
- Each wave fully coalesces64 lanes x4 B =256 B/load.
- **Shared:** all waves write the same16 KiB LDS ring.
- **Private:** each wave writes its own16 KiB LDS ring,64 KiB active total.
- Both modes allocate64 KiB LDS, so allocation/CTA residency is identical.
- Source pretraversal and200 ms selected-kernel warmup precede sampling.
  Repeated1 MiB footprint is intended to be L2-resident; no counter-based
  residency proof or physical HBM/L2 traffic measurement is claimed.

Each outer group issues64 `global_load_dword v[8:9], off ... lds` instructions
per wave, covering16 KiB. Four4 KiB instruction groups use immediate offsets
0,256,...3840 and M0 bases0/4096/8192/12288 (plus wave-private base when
selected). Immediates shift BOTH global and LDS addresses. Two vector
address instructions between M0 update and first load avoid needing a NOP.
Payload bypasses VGPRs; v8:v9 holds the source address, not loaded data.

Source offset advances16 KiB/group, wraps modulo256 KiB/wave. Across16 groups,
the four waves traverse all1 MiB, then repeat. LDS wraps each group, without
per-load destination arithmetic or a producer/consumer retirement protocol.
The opaque volatile asm owns the entire loop, M0 save/restore and clocks.

Native code: **64 loads /81 instructions per wave/group**, four M0 updates,
eight vector address instructions and five scalar offset/loop instructions.
The backedge contains **zero s_waitcnt, barriers, LDS reads/writes, checksums
or NOPs**. No register-target load, conversion, scratch/lane spills or MFMA.
No wait-depth, wave-count, ring-size or unroll sweep is added.

## Completion, timing and checking

Clock boundaries are outside the unrolled loop:

```text
initialize64 KiB LDS / join                 # outside issue interval
save M0; begin core+steady clocks
    [64 direct loads + address/loop operations] x groups
issue-end core+steady clocks
terminal VM0                              # only VM wait; outside issue interval
completion-end core+steady clocks; restore M0
join waves; copy LDS to output             # epilogue, outside both intervals
```

LGKM0 waits outside the loop make scalar clock results usable; they are not
VMEM watermarks or periodic staging throttles. The terminal VM0 makes the
subsequent LDS checker safe. No periodic barrier/completion wait was added.

CTA issue cycles = **max(issue_end)-min(begin)** across four waves; completion
uses max(completion_end)-min(begin). Clock-read overhead is included; their
difference is not isolated drain latency. Raw timestamps are retained.
Steady clock supplies us; measured core/steady ratio is ~1682.6 MHz.
Full HIP event interval additionally includes initialization/join/output and
kernel launch/completion; CPU checking/transfers are outside the event.

One four-wave load step means one64-lane DWORD instruction per wave:
**4 x64 x4 =1024 requested B**, not256 B and not4x the cycle denominator.
`requested_B_per_issue_cycle` uses all requested source payload, including
repeated source/LDS addresses, not unique destination footprint or certified
physical traffic. At default16384 groups there are1,048,576 loads/wave and
**1 GiB requested payload per sample**, from the fixed1 MiB source.

All64 lane positions have deterministic varying bits, periodic every256 B
in source. Thus writes racing/reordering onto the same LDS word have
identical values. This permits an exact final LDS check, **not validation of
which source iteration/quarter wrote last**. Address traversal is specified
by the source/ISA; the payload oracle cannot detect arbitrary incorrect
source strides that preserve lane index. Shared mode's unused48 KiB must
remain initialized sentinel; all active words and32 output guards are checked.

## Baseline result

Two counterbalanced process-order repetitions (shared/private, then private/
shared), nine checked samples/process after200 ms warmup; medians run1/run2:

| LDS destination | Core cycles/four-wave load | Requested B/core-cycle | Requested GB/s | Issue interval us |
|---|---:|---:|---:|---:|
| Shared16 KiB | 17.3966 /17.3959 | 58.8621 /58.8643 | 99.0434 /99.0449 | 10841.12 /10840.96 |
| Private4x16 KiB | 17.3962 /17.3953 | 58.8636 /58.8664 | 99.0347 /99.0581 | 10842.08 /10839.52 |

Both layouts are indistinguishable here. Full event medians are10857.58/
10854.85 us (shared) and10859.82/10856.29 us (private); completion-end us
are within0.16 us of issue-end medians, not a claim of zero pending memory.

All **36 timing samples +6 smoke checks** pass. Smoke groups1/16/17 cover
one full LDS sweep, complete source traversal and source wrap. Data, source
snapshots, native loop/metadata, fingerprints, raw clocks and commands are
in `results/base/`; distinct measured executable/code object in
`temp/lds-staging/base/`. Existing research/binaries are untouched.

## What this does / does not establish

The no-periodic-wait stream sustains ~58.86 requested B/core-cycle. Overlapping
versus private LDS destinations introduce no measurable difference here.
This is the pure-copy reference under natural backpressure, **not proof
that a specific outstanding-transaction queue/count is the only bottleneck**.
L2 returns, transport, LDS write delivery and issue resources remain coupled.

Two such per-wave loads supply the2 KiB staged input of one64x64x8 GEMM
step: ~34.8 cycles of pure-copy service at this measured rate. That is an
accounting comparison only: GEMM also reads LDS into operands, synchronizes
and executes MFMA. This probe does not measure simultaneous read/write
contention, determine its total LDS ceiling or rule out staging as a GEMM
bottleneck. No universal32/64 B/core-cycle physical roofline is inferred.

## Reproduce

```sh
bash research/lds-staging/build.sh                    # optional fresh build path
python3 research/lds-staging/inspect_code.py temp/lds-staging/base
numactl --cpunodebind=0 --membind=0 temp/lds-staging/base/probe \
  --mode shared --groups 16384 --rounds 9 --warmup-ms 200
# --mode private for the other destination layout
python3 research/lds-staging/run.py --output <fresh-results-directory>
```

Defaults:1 MiB source,64 KiB allocated LDS, four waves,64-load unroll,
16384 groups, nine samples,200 ms warmup. No production/CMake/model/system
changes, package installation, b128 work or modifications to other research.
