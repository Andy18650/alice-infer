# Fixed K addresses: repeated first fragment, all pipeline work retained

Before this intentionally non-GEMM experiment, the correct real-A W8A16 GEMMs,
grid tests and single-CTA diagnostic were committed as **`21c735b`**. The user
requests no hardware counters: remove K progression from loading addresses
and repeatedly use the first A columns/B rows to inspect the pipeline itself.

**Result:** the fixed-address body still does not reach the resident ~64-cycle
rate. With deep queues it settles near **96–99 cycles/MFMA for FP8 B** and
**~129 for FP16 B**. Warm/eviction-prepared differences nearly disappear.
Both formats' logical load-result payloads correspond to about **32 bytes
per CTA core cycle**, a strong byte-scaling clue, not a hardware bandwidth
counter or identification of a particular cache/return/issue unit.

## Exact change and retained work

This round starts with one timed CTA, four N waves, as announced. There is no
2x2 wave layout, full-grid tuning or new operand reuse. All prior source and
measured binaries remain unchanged.

`fixed_probe.hip` compiles both real-K and fixed-K controls in one binary.
For fixed K, **every priming, queue-fill and refill load uses fragment 0**:

```text
A pointer = A tile base + lane offset
B pointer = this wave's N-tile base + lane offset
```

The bases still select the proper M/N packed tiles; the current K-loop index
never advances either pointer. There is no compact scratch repack or address-
phase sweep: the original per-wave stream bases/pitches remain, so cache-bank
or register-placement effects are not ruled out merely by the small footprint.

`32x32x8f16` consumes eight K elements. This repeats the **first eight columns
of A and first eight rows of B**, not a scalar rank-one update. Expected output:

```text
C = steps * (A[:,0:8] * B[0:8,:])
```

It is deliberately no longer the full K-varying GEMM. Output contains four
32x32 tiles. Each iteration still performs one native MFMA, two real global
loads, the paired wait, A copies and (for FP8) all eight conversion VALUs.
D=1/2/4/8/16, two ready A/B banks and `vmcnt(2D-2)` are unchanged.

The assembly is `volatile` only to prevent compiler elimination/hoisting.
**It does not request volatile/uncached GPU memory:** no glc/slc/cache-bypass
flags are added. The ordinary hardware cache policy is unchanged.

Per-CTA distinct input bytes, independent of loop length:

| Input | A | B across four N waves | Total |
|---|---:|---:|---:|
| A16/B8 | 512 B | 4*256 B | **1536 B = 1.5 KiB** |
| A16/B16 | 512 B | 4*512 B | **2560 B = 2.5 KiB** |

Both are below the user's 16 KiB/CU L1 capacity and possible 2 MiB allocation-
locality L2 budget. Full arrays are still allocated for the real-K co-run
controls; those are not the fixed path's active footprint. Initial blocking
seed loads already read all fixed tiles before the core timer. The warm path
pre-reads just that fragment; the eviction-prepared path may begin cold at
launch but its priming warms the same tiny set. Their steady-state convergence
is expected. No cache hit rate or literal zero load latency is asserted.

## Actual ISA and correctness

`audit_fixed_probe.py` verifies all 20 streaming entries from the executable:

- Fixed-K A address operands use the same invariant two VGPRs at every load
  site, and likewise for B. No body instruction redefines either pointer.
- Exactly 32 MFMAs, 64 A/B global loads and 32 paired waits per backedge.
  FP8 retains exactly 64 each of permute/packed shift/mask/packed multiply.
- The raw load ring survives the backedge, with exact 2D credits and 2D-2
  waits. Three-lap pending-register safety and 32 injected early-read negative
  controls pass; the final queue is drained before endpoint reuse.
- No private/scratch/LDS storage or lane spills. The pure resident anchor
  still has only 32 MFMA vector instructions/backedge.

For fixed D8 FP8, the loop has two A-copy VALUs plus eight B-conversion VALUs
per MFMA; **the four per-step address VALUs from the variable-K counterpart
fold away**. SGPR usage drops from 68 to 20, with reported VGPR count 56 and
16 AGPRs. FP16 fixed D8 reports 64 VGPRs, 22 SGPRs, 16 AGPRs. This is not an
isolated cache-latency subtraction: address generation and register layout
also change, as the user explicitly requested by freezing the addresses.

CPU references are separately selected: full joint K-phase reduction for
variable K, and first-fragment product times repetitions for fixed K and the
resident anchor. All **1008 timed samples** and **672 independent A-only/
B-only/both-change/restore, warm-path, poisoned-output/guard cases** pass
exactly versus FP64. Fixed values alone can hide stale same-value reads;
the varying-K controls and FIFO/physical-register audit remain essential.

## Protocol and main result

Native HIP O3/C++20, gfx90a, ROCm 7.1 / Clang 21.1.8. HIP 0, one timed
256-thread CTA/four wave64s, CPU+memory NUMA node 0, **no model/context**, no
clock/power settings changed. Fixed unroll32, one timed dispatch/pass.
Two independent nine-round 4096-repetition processes (1729/2718), three rounds
each at 1024 (31415) and 8192 (16180), randomized case/cache order.

Core clocks start after readiness/queue priming and any in-dispatch pre-read;
they include the final load drain but no explicit final-MFMA completion fence.
The eviction control is a completed 256 MiB reader outside probe timing.
Event `launch_ms` includes the warm pre-read and is not used as body TFLOP/s.
The effective core rate is approximately 1.684 GHz, measured via core/steady
counters, not assumed. No cache/performance hardware counters are used.

Median **fixed-K core cycles/MFMA**, 4096 repetitions:

| B format | D | Pre-read, run 1 / run 2 | Eviction-prepared, run 1 / run 2 |
|---|---:|---:|---:|
| FP8 | 1 | 187.572 / 188.538 | 188.630 / 188.486 |
| FP8 | 2 | 103.789 / 103.424 | 103.495 / 103.349 |
| FP8 | 4 | 98.867 / 99.859 | 99.597 / 99.425 |
| FP8 | 8 | **97.224 / 96.111** | **96.356 / 96.075** |
| FP8 | 16 | 99.132 / 99.106 | 99.165 / 98.884 |
| FP16 | 1 | 173.463 / 173.977 | 173.760 / 173.988 |
| FP16 | 2 | 129.089 / 129.278 | 129.275 / 129.224 |
| FP16 | 4 | 129.311 / 129.232 | 129.086 / 129.185 |
| FP16 | 8 | **129.305 / 129.296** | **129.231 / 129.287** |
| FP16 | 16 | 129.084 / 129.297 | 129.058 / 129.276 |
| Resident MFMA | — | **63.995 / 63.995** | **63.995 / 63.995** |

Co-run variable-K D8 controls retain ~101/119 FP8 and ~170–174 FP16 intervals,
consistent with the preserved previous experiment's broad result. Use these
co-run controls, not cross-executable absolute timing, for comparisons.

### Longer and shorter repetition controls

At 8192 repetitions, fixed FP8 D8 is 96.053 cycles for both cache preparations;
D16 is 96.103/96.104. FP16 D8 is 128.816/128.594 and D16 128.548/128.660;
resident remains ~63.998. Thus the D16 penalty is not a universal constant:
its longer run converges to D8, but never recovers 64.

At 1024, fixed FP8 D8 is ~97.6/~96.3, while D16 is ~110.3/~109.2; fixed FP16
D8/D16 is ~133. These shorter controls expose finite-length/entry/code-layout
sensitivity. Their three rounds are retained, not promoted as steady-state
ceilings. Short 32/96 runs are correctness-only checks.

## Narrow interpretation

The tiny repeatedly requested set and convergence of cache preparations make
large-working-set/DRAM-miss explanations much less persuasive for the steady
fixed case. Yet **real loads are not free even when cached**: issue, delivery,
register writes and waits remain. We found no correctness defect in these
checked queue/readiness paths, but this does not prove the performance schedule
is optimal or identify a specific hardware unit.

One particularly strong clue is byte scaling across formats. Four waves
request per logical step:

```text
FP8:  4*(512 A + 256 B) = 3072 bytes
FP16: 4*(512 A + 512 B) = 4096 bytes
```

Run-2 warm D8 gives **31.96 / 31.68 logical bytes/core-cycle**; the long run
is also near 32. FP16 has no conversion work and fewer VALUs, yet is slower
roughly in the **4/3 payload ratio**. This is consistent with a CU-local load/
result-delivery throughput constraint rather than FP8 conversion or missing
DRAM-latency coverage alone. It is **not a measured physical L1 bandwidth or
proof of which port/cache/issue/dependency path limits throughput**. Address
bank/set and register placement effects remain possible.

Round concluded without counters, 2x2 wave tiling, new reuse or corrective
tuning. The real packed GEMM checkpoints are safe at `21c735b`; this new
repeated-fragment probe is deliberately labeled non-GEMM diagnostic work.

## Reproduce

```sh
bash research/fp8-w8a16/build-fixed-probe.sh
python3 research/fp8-w8a16/audit_fixed_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/fixed/fixed-probe 4096 9 1729
# Repeat: 4096 9 2718. Shorter: 1024 3 31415. Longer: 8192 3 16180.
python3 research/fp8-w8a16/analyze_fixed_probe.py
```

Raw runs, logs, flags, hashes, audit and summary are in `results/fixed/`; actual
ISA, loops, metadata and binaries are in `temp/fp8-w8a16/fixed/`. The analyzer
checks completeness, exact math and variable/fixed labels. No engine/CMake
code changes or engine-regression rerun are claimed for this standalone test.
