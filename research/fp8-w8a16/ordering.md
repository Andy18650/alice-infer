# Moving conversion after MFMA: the boundary cost persists

User-directed A/B, not an attempt to optimize away an already accepted cost.

**Outcome: the proposed phase rotation is correct, but does not remove the
constant backedge cost.** The original ordering fits about **12 cycles per
backedge**; MFMA-first/conversion-after fits about **20 cycles per backedge**.
The pure-FP16 control still runs close to its nominal 64-cycle arithmetic
rate. There is no justification to select the new ordering as a win.

## Exact change

Original steady-state loop (mode 1):

```text
next = convert(packed_weight)     // Eight VALUs, writes other bank
MFMA(current)
current = next                   // Logical bank alternation
```

New steady-state loop (mode 2):

```text
// Prologue: explicitly convert both initial ready operands.
current = convert(packed_weight)
ahead   = convert(packed_weight)

repeat:
    MFMA(current)
    current = ahead
    ahead = convert(packed_weight)
```

This is a genuine rotation of the conversion relative to the branch: the
new loop target starts with MFMA, and its last vector instructions before
the backedge are the final conversion. It retains a ready future operand;
it does **not** immediately consume the register just written by packed
multiply. Each MFMA has one in-loop eight-VALU conversion, with two extra
conversions in the prologue. The final two prepared operands are unused.
All activation and weight operands stay in registers; no LDS or timed
operand streaming. No inserted NOPs in either conversion variant.

The compiler-visible previous operand and earlyclobber constraints keep the
banks distinct; compiler-only machine scheduling barriers prevent phase
reordering. The new sequence alternates two FP16 operand banks; the initial
operands are explicitly produced by conversion, not merely borrowed from
pre-expanded weights. The pre-expanded argument remains a compiler-lifetime
input to the first conversion, not its numerical result.

## Actual code-object evidence

`audit_order_probe.py` checks all nine compiled entries:

- Modes 0/1/2 = pure FP16 / conversion-before / conversion-after, each with
  2/8/32 MFMAs per backedge and one dependent accumulator chain.
- The hot-loop vector sequence is exactly MFMA-only, (eight VALUs, MFMA), or
  (MFMA, eight VALUs), repeated U times. No bank-copy instructions.
- Mode 2 has precisely two additional conversion sequences outside the loop.
- No in-loop memory loads/stores or NOPs, zero private/scratch bytes and LDS.
- Both conversion variants report **28 VGPRs, 16 AGPRs, 20 SGPRs**; the FP16
  control reports 24 VGPRs, 16 AGPRs, 20 SGPRs. No resource-count advantage
  or spilling explains the A/B difference.
- **All nine variants still have `s_waitcnt vmcnt(0)` at the loop target.**
  The compiler did not make the proposed test moot by deleting that wait.

The key new boundary, from the executable-extracted ISA:

```asm
v_mfma_f32_32x32x8f16 a[0:15], v[2:3], v[6:7], a[0:15]
v_perm_b32       v6, 0, v8, s8
v_perm_b32       v7, 0, v8, s9
v_pk_ashrrev_i16 v6, 1, v6 op_sel_hi:[0,1]
v_pk_ashrrev_i16 v7, 1, v7 op_sel_hi:[0,1]
v_and_b32_e32    v6, s10, v6
v_and_b32_e32    v7, s10, v7
v_pk_mul_f16     v6, v6, s12
v_pk_mul_f16     v7, v7, s12
s_add_i32       s13, s13, 8
s_cmp_lt_i32    s13, s11
s_cbranch_scc1  loop
loop:
s_waitcnt vmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[2:3], v[4:5], a[0:15]
```

## Protocol and results

Native HIP O3/C++20, gfx90a, system ROCm 7.1 / Clang 21.1.8. HIP logical
device 0 (104-CU MI250X GCD), CPU and memory bound to NUMA node 0,
**no model or attention context**, no clock/power overrides.

104 CTAs x 256 threads (416 waves), matching signed FP16 activations and
signed FP8 zero/subnormal/normal weights. Same data and FP64 all-output
oracle as the NOP-window probe. Each wave repeatedly computes its 32x32
output tile with one dependent FP32 accumulator chain. The initial seed
values and per-wave computations are intentionally invariant; this is an
arithmetic schedule test, not a general GEMM.

Each sample contains eight event-timed launches; last-launch per-wave
`clock64()` intervals give the mean core cycles/MFMA. All kernels are warmed,
then all three U=8 variants condition the device for >=2 seconds. Nine
randomized rounds, two independent 4096-MFMA processes (seeds 1729/2718),
plus a nine-round 8192-MFMA process (31415). GPU runs are sequential.

Median core cycles/MFMA at 4096 MFMAs/wave:

| U: MFMAs/backedge | Pure FP16, run 1 / run 2 | Conversion before, run 1 / run 2 | Conversion after, run 1 / run 2 |
|---:|---:|---:|---:|
| 2 | 64.130 / 64.128 | 70.149 / 70.148 | 74.031 / 74.031 |
| 8 | 64.130 / 64.130 | 65.648 / 65.646 | 66.531 / 66.531 |
| 32 | 64.131 / 64.130 | 64.523 / 64.522 | 64.656 / 64.656 |

The new ordering is slightly slower, including event throughput: at U=8,
run 1 is 142.406 -> 140.556 TFLOP/s, with effective rates ~1416 / ~1418 MHz.
The negative result is not explained by a lower effective clock for the
new schedule in that comparison. These remain standalone arithmetic rates,
not a model or general tiled-GEMM result.

### Separating the boundary from endpoint offsets

Fit the U=2 and U=32 medians to

```text
cycles_per_MFMA(U) = A + B/U
```

and hold U=8 out as an independent check. B is the inferred per-backedge
cost, while A includes the common arithmetic floor plus finite-length
prologue/timestamp/exit effects.

| Run | Before: A | Before: B, cycles/backedge | After: A | After: B, cycles/backedge |
|---|---:|---:|---:|---:|
| 4096, process 1 | 64.147866 | 12.002579 | 64.031316 | 20.000005 |
| 4096, process 2 | 64.147080 | 12.000851 | 64.031332 | 19.999945 |
| 8192 | 64.073910 | 11.999857 | 64.015638 | 20.000035 |

The held-out U=8 residual is <0.002 cycles/MFMA in every case. Doubling the
loop length approximately halves A's offset from 64 but does not remove or
materially alter B. Consequently, simply subtracting the pure-FP16 median
and multiplying by U is less clean than this fit: the endpoint effects are
not identical between the two phase orderings. End timestamps do not
explicitly drain the final MFMA, as in the existing probes.

All **243 timed samples** (three processes x nine cases x nine rounds)
pass every output check, with **zero observed absolute error** versus FP64.
One sample checks the last of its eight launches. The original probe and
NOP-sweep sources/executables are unchanged and still match their hashes.

## Honest interpretation / stopping point

The tested phase rotation **does not hide the exposed boundary cost**;
the fitted constant changes from ~12 to ~20 cycles. Moving the conversion
really happened in the emitted code, the target wait remains, and correctness
passes. This is a negative result for the proposed solution, not a failed
implementation or an omitted prologue.

It also does **not** tell us what the redundant-looking wait is actually
waiting for or assign the cost to a specific instruction. The pure-FP16
control has the same target wait but not the same exposed boundary cost.
Our earlier attribution to branch/loop-entry issue interaction remains a
hypothesis, not proof of a VMEM wait or accumulator-forwarding mechanism.
No unexplained eight-cycle difference is relabeled as an established
hardware latency.

Keep the accepted original ordering as the reference. The user explicitly
accepts this small cost; no further bubble-chasing, production dispatch
change, or broad microarchitecture campaign is implied by this experiment.

The subsequent [four-bank/staged-VALU diagnostic](dependency.md) uses a
fully primed generalized ring, which changes load-wait/hazard code generation.
Its absolute two-bank after constant differs from this preserved probe;
that difference is documented rather than misattributed to buffer count.

## Reproduce

```sh
bash research/fp8-w8a16/build-order-probe.sh
python3 research/fp8-w8a16/audit_order_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/order/order-probe 4096 9 1729
# Repeat at seed 2718; endpoint control: 8192 9 31415.
python3 research/fp8-w8a16/analyze_order_probe.py
```

Stdout is CSV and stderr records protocol/validation. Raw runs, flags,
source/executable/object hashes, audit and fit summaries are retained under
`results/order/`. Binaries, full ISA, individual hot loops and metadata are
under `temp/fp8-w8a16/order/`. The analyzer reads the retained runs; replacing
them is deliberate. No engine/CMake changes were made; engine regression
results are unchanged from the main research README, not newly rerun here.
