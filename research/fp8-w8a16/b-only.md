# Resident A, B-only loads: near-MFMA-rate execution at logical K32768

User requests removing A loading and returning to an in-register A operand,
keeping B loading and logical K=32768. This remains a **performance-only,
fixed-address diagnostic**. No numerical tests or output readback run.

**Result:** FP8 B-only D8 reaches **64.391 cycles/MFMA** in both independent
no-wait runs, versus fresh A+B controls **96.114/96.109**. FP16 B-only D8 is
**64.653/64.661**, versus **128.965/128.939**. The resident anchor is ~64.12.
The explicit-wait B-only D8 variant is also near 64: FP8 64.395/64.395,
FP16 64.864/64.881. Thus the near-MFMA result is not contingent on deleting
the steady-state wait once the B queue is deep enough.

## Exactly what changed

`b_only_probe.hip` is a standalone derivative of `wait_probe.hip`; all old
sources, executables and measured code objects remain untouched.

- A is loaded into a single persistent H4 before timing and used by every
  MFMA. The optional pre-read also touches A outside timing.
- No A load is issued during raw-ring priming or inside the timed loop.
- The raw A ring, two alternating ready A banks and **two per-step A-copy
  `v_mov_b32` instructions are removed**. A is never rewritten in the loop.
- B retains the same fixed fragment-zero addresses, raw ring depth
  1/2/4/8/16, two ready banks, conversion/copy sequence and refill schedule.
- No-wait mode remains primary. A waited build is retained as a control:
  one B request per step means partial waits become **`vmcnt(D-1)`**, not
  `vmcnt(2D-2)`. Prologue completion and final B drain remain.
- Four waves still cover N, with one timed CTA, unroll32, 4096 MFMA steps
  (logical K32768). There is no new tiling, packing, reuse of loaded B,
  per-case warm-up, clock override or production integration.

`inspect_b_only_probe.py` inspects the actual executable-extracted loops:
32 MFMAs and **32 B global loads** per backedge; a constant A source register
pair with no body definitions; constant B address; 64 each of the four FP8
conversion opcodes (eight VALUs/MFMA). FP8 has **zero body `v_mov_b32`**;
FP16 retains its two B copies/MFMA. Waited and no-wait bodies have identical
instructions/registers after removing waits and normalizing the branch.
This is instruction inspection, not a numerical or pipeline-safety test.

For no-wait FP8 D8, the actual loop begins:

```asm
v_mfma_f32_32x32x8f16 a[0:15], v[2:3], v[6:7], a[0:15]
global_load_dword v17, v[4:5], off
v_perm_b32 v6, 0, v10, s0
v_perm_b32 v7, 0, v10, s1
v_pk_ashrrev_i16 v6, 1, v6 op_sel_hi:[0,1]
v_pk_ashrrev_i16 v7, 1, v7 op_sel_hi:[0,1]
v_and_b32_e32 v6, s2, v6
v_and_b32_e32 v7, s2, v7
v_pk_mul_f16 v6, v6, s3
v_pk_mul_f16 v7, v7, s3
v_mfma_f32_32x32x8f16 a[0:15], v[2:3], v[8:9], a[0:15]
```

`v[2:3]` is now resident A; `v[4:5]` is the B pointer. The new B request
writes v17, while an older raw B in v10 is converted for future consumption.
The full 323-instruction no-wait D8 loop is under
`temp/fp8-w8a16/b-only/unwaited/loop-1-1-8-32.txt`.

## Measurement and results

Native HIP O3/C++20, gfx90a, HIP device 0 (AMD-SMI GPU2, BDF31:00.0),
CPU+memory NUMA node 0, no model/context in the probe, one 256-thread CTA,
no profiling counters. AMD-SMI before/after shows no other GPU processes.
Effective core frequency is ~1.684 GHz. Two nine-round processes per mode/
implementation, seeds 1729/2718; process mode/implementation order reversed
between repeats. A+B controls are freshly rerun, not just historical values.
Both in-dispatch pre-read and completed 256 MiB eviction preparations run.
Initial loads/pre-read precede core timestamps; final raw-B drain is included.

Median **B-only, no-wait core cycles/MFMA**, pre-read, run1 / run2:

| D | FP8 B | FP16 B |
|---:|---:|---:|
| 1 | 67.492 / 67.467 | 64.813 / 64.827 |
| 2 | 67.125 / 67.163 | 64.829 / 64.814 |
| 4 | 67.272 / 67.162 | 64.646 / 64.645 |
| 8 | **64.391 / 64.391** | **64.653 / 64.661** |
| 16 | 67.410 / 67.343 | 64.839 / 64.825 |

Eviction-prepared D8 pooled medians are FP8 **64.391** and FP16 **64.660**,
versus fresh A+B **96.099/128.892**. There is no relevant preparation gap.

With waits restored, shallow B queues still show latency penalties: D1 FP8
~164 cycles, FP16 ~149; D2 ~82/~76. D4 reaches ~67/~65, and D8 ~64.4/~64.9.
Full waited/no-wait timing tables and all raw cases are retained.

The inherited warm-up still favors FP8 D8 and the resident anchor. The
~67-cycle other-depth FP8 numbers are not proof of a different steady-state
slope; the earlier shallow-K study already exposed their larger finite-entry
cost. K is deliberately held at the requested 32768 in this round.

## Working model

Per four-wave logical step, the request payload changes as follows:

| Path | A+B/B-only logical bytes | At fitted 32 B/core-cycle | With 64-cycle MFMA floor |
|---|---:|---:|---:|
| A+B, FP8 B | 3072 | 96 cycles | 96 |
| A+B, FP16 B | 4096 | 128 cycles | 128 |
| Resident A, FP8 B | 1024 | 32 cycles | **64** |
| Resident A, FP16 B | 2048 | 64 cycles | **64** |

The measured ~96/~129 -> ~64/~65 transition agrees with that previously
fitted throughput model without changing its effective rate. FP8 conversion
plus one B load per MFMA fits near the MFMA rate in this fixed-address path.

**This removes both A loads and A-copy VALUs**, and changes register allocation
and code layout. It does not isolate their individual costs, identify a
particular physical cache/issue/return/register port, or measure literal L1/HBM
traffic. It is not a claim that streaming real B from HBM reaches this rate.
No valid-GEMM or numerical-correctness claim is made for the unsafe reads.

## Reproduce

```sh
bash research/fp8-w8a16/build-b-only-probe.sh
python3 research/fp8-w8a16/inspect_b_only_probe.py
numactl --cpunodebind=0 --membind=0 \
  temp/fp8-w8a16/b-only/unwaited/b-only-probe 4096 9 1729
# Repeat seed 2718, and use /waited/ for the partial-wait control.
python3 research/fp8-w8a16/analyze_b_only_probe.py
```

`results/b-only/` contains eight timing CSV/log pairs, before/after AMD-SMI,
four implementation/mode flag/hash records, instruction inventory and timing
summaries. Executables, HSACO, metadata/ISA and extracted loops remain under
`temp/fp8-w8a16/b-only/{waited,unwaited}/`. Numerical testing stays OFF.
