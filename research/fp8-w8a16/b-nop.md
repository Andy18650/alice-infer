# MFMA -> NOP: the payload-only prediction is only a partial match

User asks to test the hypothesis that replacing B-only MFMA with NOP would
expose ~32 cycles/step for FP8 B and ~64 for FP16 B. This follows the explicit
qualification that **B preparation remains**: eight FP8 conversion VALUs or
two FP16 B copies, plus the NOP's issue/delay cost. It is not pure load-only.

**Observed at D8, no steady wait:** FP8 **45.394/45.394 cycles/step**, not 32;
FP16 **64.601/64.592**, close to 64. This is a **partial match, not validation
of a complete payload-only performance model**. A fitted 32 logical B/core-
cycle memory ceiling remains consistent with FP16; the FP8 loop has additional
work/effects not isolated by this experiment. Do not assign its entire excess
to conversion or identify a physical port from these results.

## Exact ablation

`b_nop_probe.hip` derives from the preserved `b_only_probe.hip`. Its streaming
kernel replaces each MFMA with:

```cpp
asm volatile("s_nop 0" : "+a"(c) : "v"(a), "v"(b[j%2]) : "memory");
```

The dummy inputs/output preserve compiler-visible accumulator/operand
lifetimes. **They do not cause hardware reads/writes at the NOP**. That is
the intended removal of MFMA work. All loads, B conversion/copies, raw ring,
ready banks, scheduling barriers, loop unroll and endpoint drain remain.
The separate MFMA-only resident anchor is deliberately unchanged.

`inspect_b_nop_probe.py` extracts all streaming loops from the executable
code objects. In every format/depth/wait mode it finds exactly 32 `s_nop 0`
and no MFMA in the loop. After replacing each compute slot by a placeholder
and normalizing branch displacements, **all other instructions AND register
assignments are identical to their corresponding MFMA controls**. FP8 still
has 32 dword B loads and 256 conversion VALUs per backedge; FP16 has 32
wide B loads and 64 B-copy VALUs. No A loads or copies appear. The NOP is
not claimed to be zero-cost; no extra NOP calibration/tuning is done here.

The no-wait FP8 D8 loop begins:

```asm
s_nop 0
global_load_dword v17, v[4:5], off
v_perm_b32 v6, 0, v10, s0
v_perm_b32 v7, 0, v10, s1
v_pk_ashrrev_i16 v6, 1, v6 op_sel_hi:[0,1]
v_pk_ashrrev_i16 v7, 1, v7 op_sel_hi:[0,1]
v_and_b32_e32 v6, s2, v6
v_and_b32_e32 v7, s2, v7
v_pk_mul_f16 v6, v6, s3
v_pk_mul_f16 v7, v7, s3
```

Full loops: `temp/fp8-w8a16/b-nop/{waited,unwaited}/loop-*.txt`.

## Timing protocol

Same logical K32768 / 4096 steps, unroll32, one 256-thread CTA/four N waves,
fixed fragment-zero B, resident A, native HIP O3/C++20/gfx90a, HIP0
(AMD-SMI GPU2/BDF31:00.0), CPU+memory NUMA0, no model/context in the probe,
no clock/power changes or profiling counters. AMD-SMI before/after shows no
other GPU processes. Effective core clock is approximately 1.684 GHz.

Two nine-round processes per implementation/wait mode, seeds 1729/2718.
Implementation/mode process order reverses between repeats; case/cache order
is randomized. The MFMA control is freshly rerun. Both pre-read and completed
256 MiB eviction preparation run. The inherited two-second warm-up favors
FP8 D8 and the resident anchor; other-depth finite-length caveats remain.
Initial pre-read/seed/queue issue precedes the core interval, final raw-B
drain remains inside it. No backedge optimization or further ablation.

**No numerical checks or output readbacks run.** The NOP CSV column is named
`cycles_per_step`, not `cycles_per_mfma`; the comparison analyzer understands
both names. Do not report FLOP throughput for the NOP loop.

## Results

Median core **cycles/step**, pre-read, run1 / run2:

| Format / D | MFMA, no wait | NOP, no wait |
|---|---:|---:|
| FP8 / 1 | 67.330 / 67.252 | 48.217 / 48.304 |
| FP8 / 2 | 67.045 / 67.293 | 48.223 / 48.419 |
| FP8 / 4 | 67.307 / 67.151 | 48.387 / 48.382 |
| FP8 / 8 | **64.391 / 64.391** | **45.394 / 45.394** |
| FP8 / 16 | 67.360 / 67.466 | 47.642 / 47.675 |
| FP16 / 1 | 64.703 / 64.867 | 64.618 / 64.575 |
| FP16 / 2 | 64.807 / 64.759 | 64.515 / 64.508 |
| FP16 / 4 | 64.648 / 64.851 | 64.638 / 64.642 |
| FP16 / 8 | **64.660 / 64.771** | **64.601 / 64.592** |
| FP16 / 16 | 64.639 / 64.707 | 64.463 / 64.471 |

The eviction-prepared NOP D8 pooled medians are FP8 **45.395**, FP16
**64.497**. With partial waits restored, NOP D8 FP8 is **48.900/48.901**,
FP16 **64.647/64.794**. All shallow waited cases and both preparations are
retained in raw CSV/summary artifacts rather than discarded.

## Hypothesis verdict

The logical request counts are unchanged: 1024 B/step for four-wave FP8 B,
2048 B/step for FP16 B. Dividing by the hypothesized effective rate 32 gives
32 and 64 cycles. **Only the FP16 whole-loop prediction matches.**

The FP8 result does not falsify the possibility of a 32 B/core-cycle memory
service ceiling: its eight retained conversion VALUs, NOP/issue costs,
dependencies and possible interactions can set another floor. But this
experiment does not isolate those components, so that explanation remains a
hypothesis, not an established cause of the ~45.4 interval. Conversely,
FP16 matching does not uniquely identify a hardware port or prove that 32 is
a universal independent memory limit. No cache/NoC/HBM traffic was measured.

A complete model must account for more than logical payload. No conversion-
removal, NOP-removal or other extra test is performed in this round.

## Reproduce and artifacts

```sh
bash research/fp8-w8a16/build-b-nop-probe.sh
python3 research/fp8-w8a16/inspect_b_nop_probe.py
numactl --cpunodebind=0 --membind=0 \
  temp/fp8-w8a16/b-nop/unwaited/b-nop-probe 4096 9 1729
# Repeat 2718; /waited/ retains partial waits.
python3 research/fp8-w8a16/analyze_b_nop_probe.py
```

`results/b-nop/`: eight timing CSV/log pairs, snapshots, four implementation/
mode flag/hash records, instruction inventory, summaries. Executables,
HSACO/ISA/metadata and extracted loops are under `temp/fp8-w8a16/b-nop/`.
All previous sources and binaries are preserved, with no production changes.
