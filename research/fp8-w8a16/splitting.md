# Preheader isolation + explicit split conversion: boundary term removed

User-approved two-part experiment. **The plan works in this register-only
prototype.** Explicit preheader initialization/synchronization removes the
repeated loop-head wait. Keeping 8/8 conversion placement still leaves a
~4-cycle/backedge term in the primary two-bank case; explicit **10/6** (and
9/7) placement removes that term at measured precision.

This reaches **~64.009 core cycles/MFMA**, matching the raw-FP16 arithmetic
rate, but not identical event TFLOP/s: FP8 conversion is ~145 versus ~150
TFLOP/s, **96.54% / 96.53%** in the two matched runs, consistent with the
lower effective core-clock rate. It remains a register-resident arithmetic
prototype, not a streaming-weight or general-GEMM result.

## 1. Preheader experiment, unchanged 8/8 body

`split_probe.hip` provides raw FP16 and explicitly enumerated MFMA-first
conversion bodies. The primary two-bank 8/8 control reproduces the prior
order probe's ~20-cycle boundary constant, including the target
`s_waitcnt vmcnt(0)`.

Preheader variants:

| ID | Explicit work before core-clock body |
|---:|---|
| 0 | Original compiler-managed initialization/load waits |
| 1 | Force initialized FP32 AGPR accumulator + operand readiness, explicit `s_waitcnt vmcnt(0)` |
| 2 | ID 1 + `s_nop 0` (one encoded wait unit) |
| 3 | ID 1 + `s_nop 3` |
| 4 | ID 1 + `s_nop 7` |
| 5 | ID 1 + two independent scratch XOR VALUs |
| 6 | ID 1 + four independent scratch XOR VALUs |
| 7 | ID 1 + eight independent scratch XOR VALUs |

An inline-asm **`+a` accumulator constraint** forces the initial accumulator
into AGPRs before this preheader block. It does not modify the accumulator;
the GPU values remain zero. The compiler emits one AGPR zero write plus
fifteen AGPR moves in the explicit-preheader variants. Readiness is enforced
by a real wait, not by hoping a fixed NOP delay covers a global load. The
operand inputs and compiler scheduling barrier keep the block before timing.

The actual-code-object audit verifies:

- ID 0 has **one in-loop `s_waitcnt vmcnt(0)`** in all entries.
- Every explicit-preheader variant, IDs 1..7, has **no in-loop wait or NOP**.
- Accumulator initialization, explicit wait, requested NOP/XOR work all occur
  **before the beginning core timestamp**. GPU-event timing still includes them.

For the primary two-bank 8/8 body, median core cycles/MFMA at U=8:

| Variant | Run 1 | Run 2 | Fitted boundary cost B, run 1 / run 2 |
|---|---:|---:|---:|
| Original preheader | 66.531316 | 66.531344 | 20.000 / 20.000 |
| Explicit init + wait only (ID 1) | 64.509780 | 64.509780 | 4.000 / 4.000 |
| ID 1 + `s_nop 7` (ID 4) | 64.513686 | 64.513681 | 4.000 / 4.000 |
| Raw FP16 with ID 1 | 64.009787 | 64.009780 | ~0 / ~0 |

This strongly supports the user's observation that the repeated header
instruction was a first-entry/prologue requirement. It does **not** prove
that the wait itself has an architectural 16-cycle latency: initialization,
wait placement and the resulting machine-code layout changed together.

## 2. Explicit per-interval redistribution

The loop body is written out using explicit `STEP(0)`, `STEP(1)`, ... calls,
not an automatically unrolled uniform inner loop. U=4/8/32 uses a common
explicit prefix and a separately written last pair. There is still a runtime
outer backedge and one dependent FP32 accumulator chain.

Baseline end of body:

```text
penultimate MFMA
full conversion: 8 VALUs
final MFMA
full conversion: 8 VALUs
counter / compare / branch
```

The 10/6 version moves both byte permutations of the final conversion earlier:

```text
penultimate MFMA
full conversion: 8 VALUs
final conversion: 2 byte permutations
final MFMA
final conversion: 2 packed shifts + 2 masks + 2 packed multiplies
counter / compare / branch
```

Total work is unchanged: **eight conversion VALUs per issued MFMA**. Other
points move one/three/four leading instructions, making 9/7, 11/5, 12/4.
No implementation moves a whole extra conversion into the preceding interval.

### Register lifetime and actual instruction stream

Early partials must not clobber the final MFMA's still-live source. The
begin-conversion assembly uses earlyclobber outputs and a compiler-visible
keep-live operand. The allocator redirects an earlier conversion's ready
result to another pair when needed, freeing the partial-conversion pair
without inserting a data-copy instruction. It is not assumed to use just
two physical pairs throughout the body, despite the logical two-bank queue.

`audit_split_probe.py` checks all **168 compiled entries** in the actual
executable-extracted gfx90a object:

- Exactly U MFMAs, 8*U conversion VALUs, and no copy VALUs or in-loop memory.
- Per-MFMA VALU interval counts are exactly `[8,...,8,10,6]` for the 10/6
  body, and corresponding 9/7, 11/5, 12/4 sequences for the other splits.
- Each conversion register advances permuted -> shifted -> masked -> ready.
  Three cyclic iterations are checked, so partial values may never be read
  as MFMA operands or overwrite activation registers, including at wraparound.
- Preheader placement and all scalar loop instructions are recorded.
- Zero private/scratch bytes and zero LDS. Primary two-bank controls and
  splits report **28 VGPRs, 16 AGPRs, 20 SGPRs**, without increasing the
  reported register count for 10/6. Raw ID-1 FP16 reports the same counts.

The source uses explicit calls plus scheduling constraints, not a single
monolithic assembly loop. Code-object verification, rather than source
appearance, establishes that the intended schedule survived compilation.

## Primary results: two-bank schedule

Median core cycles/MFMA with the **same explicit init/wait-only preheader
(ID 1)** on every row, U=8:

| Body | Run 1 | Run 2 | Fitted B, cycles/backedge |
|---|---:|---:|---:|
| Raw FP16 | 64.009787 | 64.009780 | ~0 |
| 8/8 | 64.509780 | 64.509780 | 4.000 |
| 9/7 | 64.009789 | 64.009791 | ~0 |
| **10/6** | **64.008801** | **64.008798** | **~0** |

11/5 and 12/4 also have approximately zero B under ID 1. The tiny differences
below a hundredth of a cycle between the zero-B rows are endpoint effects,
not a claim to beat the nominal 64-cycle arithmetic rate.

The 8192-MFMA control gives 64.004893 raw versus 64.004400 for 10/6, again
B approximately zero; endpoint offsets approximately halve. Fits use
`C(U)=A+B/U` at U=4/32 and hold U=8 out. Maximum held-out residual over all
valid fits in the experiment is **0.001136 cycles/MFMA**.

With the original ID-0 preheader left intact, partial splitting alone only
reduces the two-bank boundary term: 8/8 -> 20, 9/7 -> 16, 10/6 -> 12,
11/5 -> 8, 12/4 -> 4 cycles/backedge. The clean primary win needs both parts
of the user's plan.

### Effective clock and event throughput are not identical

For the primary ID-1 comparison at U=8:

| Work | Run 1 TFLOP/s | Run 2 TFLOP/s | Approximate effective core MHz, run 1 |
|---|---:|---:|---:|
| Raw FP16 | 150.190 | 150.384 | 1460 |
| FP8 10/6 | 144.995 | 145.161 | 1410 |

Net event throughput is **96.54% / 96.53%** of matching raw FP16. Event timing
includes priming, initialization, preheader padding, timestamps, output and
launch effects; the core-clock interval excludes explicit preheader work.
The lower effective clock is consistent with the remaining wall-time gap;
no unique power-policy or firmware cause is established or changed. The user
confirms that lower clocks under additional math load are expected on this
chip and accepts the remaining ~3.5% gap; it is not an optimization target.

## Other variants and negative results retained

Do not generalize this into “any pad fixes everything.” All tested variants
are retained in raw data and the analyzer, including less favorable ones:

- Two-bank 8/8 + four independent preheader XORs (ID 6) has no in-loop wait,
  but fits B~24 rather than 4. Other padded two-bank 8/8 cases fit B~4.
- A separate **four-bank lagged-target** control makes a future target pair
  dead before the last MFMA: step j consumes bank j and converts bank j-1
  modulo four. It is not the previous dependency study's convert-current
  ring. Under ID 4 it also has 8/8 B~4 and split 9/7 or 10/6 B~0.
- That four-bank schedule under IDs 1/7 instead has 8/8 B~24; 10/6 only
  reduces it to ~12. ID 0 happens to give 8/8 B~8 and 10/6 B~0.

Thus removal of the repeated wait is real, but not a complete explanation
for every code variant. Prologue changes also alter layout/register choices;
the remaining non-monotonic constants have **not** been assigned to a
specific hardware mechanism. The recommended research reference from this
experiment is the simple **two-bank + explicit init/wait + 10/6** case,
not a claim that padding duration alone determines the result.

## Measurement / correctness scope

Native HIP O3/C++20, gfx90a, system ROCm 7.1 / Clang 21.1.8. One HIP-0
104-CU MI250X GCD, CPU+memory NUMA node 0; **no model or attention context**,
no clock/power overrides. 104 CTAs x 256 threads (416 waves); this is nominal
four waves/CU, not a measured assignment trace. Activations and weights are
register-resident; no future-weight load or LDS pipeline is measured.

Warm every entry, then condition raw/8-8/10-6 U=8 variants for >=2 seconds.
Two independent nine-round processes at 4096 MFMAs/wave, seeds 1729/2718;
a three-round 8192-MFMA process, seed 31415. Each sample contains eight
HIP-event-timed launches, with per-wave core/steady-clock intervals from
the final launch. GPU measurements are sequential. End timestamps do not
explicitly drain the final MFMA; this is a sustained issue-rate measurement.

All outputs of the final launch in **all 3528 timed samples** match the
independently indexed FP64 reference **exactly**; no variant is excluded for
incorrect numerics. The signed dyadic input matrices include FP8 zeros,
subnormals and normals; the primitive's exhaustive conversion tests remain
in the original probe. This is not independent random-weight GEMM coverage
or an engine canonical test. Original benchmark sources and binaries remain
unchanged; no production engine/CMake changes are made.

## Reproduce / artifacts

```sh
bash research/fp8-w8a16/build-split-probe.sh
python3 research/fp8-w8a16/audit_split_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/split/split-probe 4096 9 1729
# Repeat: 4096 9 2718. Length control: 8192 3 31415.
python3 research/fp8-w8a16/analyze_split_probe.py
```

Retained stdout CSV/stderr logs, flags, hashes, audit, medians and fits are
under `results/split/`. Full ISA, preheader code, extracted loops, metadata
and binaries are under `temp/fp8-w8a16/split/`. The analyzer reads the retained
runs and verifies completeness/exact correctness before fitting. Engine
regressions were not rerun for this isolated experiment; previous engine
results remain those in the main research README.
