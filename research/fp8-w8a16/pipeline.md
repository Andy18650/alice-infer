# Load-to-convert depth 2/4/8: strong improvement, A kept free

User-directed next step after the immediate-wait baseline. A remains the
register-resident stand-in for arbitrarily fast activation supply, as requested.
**Only the weight load-to-convert pipeline is deepened.** No A loading, M/N
weight reuse, grid search, LDS sharing, partial-conversion redistribution or
backedge tuning is added.

**Result:** FP8 depths 1/2/4/8 reach approximately **18.9 / 35.2 / 62.2 /
86.3 TFLOP/s** in process 1 at U=8. Process 2 confirms the trend. Depth 8 is
**4.56x / 4.55x** the co-run depth-1 baseline, at ~1.35/~1.34 TB/s useful
weight payload. Its mean per-wave interval is ~117/~116 core cycles/MFMA,
not yet the register-only ~64-cycle arithmetic rate.

## Queue definition and steady-state schedule

Depth D means D issued load credits before each consumption wait, **not D
converted FP16 operand banks**. There are still two ready MFMA operand banks.

Prologue, before the core-clock interval:

1. Load/convert logical fragments 0 and 1 into the two ready operand banks.
2. Establish activation/accumulator readiness and drain those initial loads.
3. Issue D-1 raw weight loads for fragments 2..D, without waiting for them.

Step s, with head = s%D and tail = (s+D-1)%D:

```text
MFMA(activation, ready[s%2])
issue raw load for fragment s+D+1 into free raw slot tail
wait vmcnt(D-1)                    // retire the oldest, not all requests
convert oldest raw slot head into ready[s%2]
```

The newly converted value is consumed at logical step s+2. For D=1, the
issue/wait/convert order reduces to the immediate-wait baseline. For larger
D, that raw fragment was loaded D-1 iterations earlier. The raw ring remains
live across every outer backedge; there is no batch drain/re-prime per unroll.

| D | Raw loads primed | Steady wait | Pending credits after wait |
|---:|---:|---|---:|
| 1 | 0 | `vmcnt(0)` | 0 |
| 2 | 1 | `vmcnt(1)` | 1 |
| 4 | 3 | `vmcnt(3)` | 3 |
| 8 | 7 | `vmcnt(7)` | 7 |

This is a program-level credit bound, not a hardware-counter measurement of
physical requests in flight: loads can complete before their wait.

At kernel exit, one explicit full drain resolves unused trailing requests
**before** their registers are reused for timestamps/output. The core interval
includes this final drain; GPU-event timing additionally includes priming,
initialization, queue fill and output. The 8192-step control checks sensitivity
to finite-length endpoints. No drain is inserted inside the D>1 body.

### Correct asynchronous-register use

The first draft used a separate wait with a `+v` operand. LLVM legally copied
a still-pending FP16 value into another register **before** the wait, since
it cannot infer asynchronous load readiness from opaque inline assembly.
The changed-input oracle caught incorrect outputs at depth 8; that draft is
not included in the reported performance data.

The final implementation puts the wait and first data read in **one assembly
block**:

- FP8: wait directly followed by the permutation/shift/mask conversion. No
  extra copy VALUs are introduced in the FP8 path.
- FP16 control: wait followed by two explicit dword copies into a ready
  operand pair. Those two VALUs are retained/accounted for at every depth;
  this control is not falsely described as a zero-copy stream.

The steady-state VMEM class is uniform within a kernel (dword FP8 loads or
paired-dword FP16 loads); initial mixed activation/seed loads are fully drained
before raw-ring priming. This is validated gfx90a behavior, not a portable
claim about arbitrary mixed-operation partial waits.

## Same data and layout at every depth

The MFMA-packed layout is unchanged:

```text
[wave][K-step][lane][four FP8 bytes]
```

Each wave reads a contiguous 256-byte FP8 tile, or a corresponding 512-byte
FP16 tile. Waves have disjoint streams. The 17-pattern, 289-step phase and
changed-input epochs are unchanged from `loads.md`; strict FP64 oracles and
poisoned output guards check every output.

All depths share **the same `steps+10` records per wave**, rather than changing
row stride with D. This safely accommodates the deepest lookahead while
keeping addresses/packing matched within the sweep. At 4096 steps the FP8 /
FP16 allocations are **417.015625 / 834.03125 MiB**; at 8192 they are
833.015625 / 1666.03125 MiB.

Each loaded kernel requests `steps + D + 1` fragments/wave including priming;
only the first `steps` are consumed by MFMA. Requested/useful weight-byte
ratio is `(steps+D+1)/steps`. One padding record remains unused even at D=8.
Both bandwidth fields are shader-requested payload rates, **not physical
HBM traffic or a bandwidth-ceiling measurement**.

Depth 1 is the matched baseline **in this harness**. The old immediate-wait
probe uses a different `steps+2` row stride, and the new FP16 queue control
has explicit safe copies. Therefore speedups use the co-run D1 result, not
the earlier ~18.5 TFLOP/s result. Earlier programs/results remain unchanged.

## Actual-code-object audit

`audit_pipeline_probe.py` verifies all 20 entries from the executable:

- U=8/32 MFMAs per outer backedge, D=1/2/4/8 for streamed FP8/FP16, plus the
  resident FP16 and ordinary 8/8-FP8 controls.
- One new weight load per MFMA; the U steady waits are exactly `vmcnt(D-1)`.
  No accidental full waits inside D>1 loops.
- Prologue raw credits are D-1; peak issued credits are D; credits after each
  body remain D-1. Three cyclic iterations are simulated pessimistically,
  assuming nothing completes until a wait retires it.
- No register read **or write** may touch an unresolved load destination.
  This includes address calculations and compiler-inserted phi copies. Twelve
  deliberately injected premature-read negative controls are rejected.
- Exact FP8 conversion counts, explicit final full drain before end timestamps,
  no in-loop stores/LDS, no private/scratch bytes and no SGPR lane spills.

U=8 resource metadata (all use 16 AGPRs):

| Format | D | VGPRs | SGPRs |
|---|---:|---:|---:|
| FP8 | 1 | 28 | 19 |
| FP8 | 2 | 32 | 19 |
| FP8 | 4 | 32 | 20 |
| FP8 | 8 | 36 | 22 |
| FP16 | 1 | 32 | 18 |
| FP16 | 2 | 32 | 18 |
| FP16 | 4 | 36 | 18 |
| FP16 | 8 | 44 | 21 |

These are metadata counts, not measured occupancy. No dispatch is selected
from U=8/32 differences; unroll is a diagnostic and the backedge is untouched.

## Matrix shape / what the benchmark represents

At the primary 4096-step setting, every wave computes a **32x32 output tile**
with effective **K=8*4096=32768**. All 416 waves traverse their complete K
range; there is no split-K or partial-output reduction. A 256-thread CTA has
four waves, corresponding to four independent 32-column tiles.

Because every wave uses the same 32 activation rows and a disjoint weight
column tile, the whole grid can be viewed as:

```text
A: 32 x 32768
B: 32768 x 13312        // 416 waves * 32 columns
C: 32 x 13312
```

Equivalently, it is 416 per-wave 32x32x32768 GEMMs sharing A. Each CTA covers
32x128 logical output elements. The logical FP8 B payload is 416 MiB; the
common lookahead padding increases allocation to 417.015625 MiB.

**A is not a general streamed 32x32768 matrix:** its 32x8 fragment is repeated
at every K step and supplied from registers, under the user's free-A model.
No row/column dimensions are padded in the reported FLOP count. The 8192-step
control doubles effective K to 65536, retaining M=32 and N=13312.

The user identifies ~1.38 TB/s as this GCD's practical streaming ceiling and
notes that the queue/conversion loop may be useful for a future FP8 W8A16
GEMV. Retain it as a reference candidate, **not a validated M=1 GEMV kernel**.
For precision, the ~1.38 TB/s result belongs to the FP16 control; the FP8
path here reaches ~1.34–1.35 TB/s. No separate ceiling experiment is added.

## Measurements

Native HIP O3/C++20, gfx90a, ROCm 7.1 / system Clang 21.1.8. HIP device 0,
104-CU MI250X GCD, CPU+memory NUMA node 0, **no model or attention context**,
no clock/power overrides. 104 CTAs x 256 threads = 416 waves (nominal four
waves/CU, not a measured assignment trace). Each wave uses one dependent
FP32 accumulator chain; A is already in registers and repeated freely.

Warm all entries, then condition all U=8 cases for >=2 s. Two independent
nine-round 4096-step processes (seeds 1729/2718), plus three rounds at 8192
(31415). Randomized case order; eight event-timed launches/sample; last-launch
per-wave core/steady intervals. All runs are sequential on the GPU.

### FP8, U=8

| D | Cycles/MFMA, run 1 / run 2 | TFLOP/s, run 1 / run 2 | Useful weight GB/s, run 1 / run 2 |
|---:|---:|---:|---:|
| 1 | 569.885 / 574.027 | 18.934 / 18.859 | 295.847 / 294.680 |
| 2 | 298.923 / 299.249 | 35.155 / 35.144 | 549.298 / 549.131 |
| 4 | 163.349 / 163.189 | 62.229 / 62.235 | 972.333 / 972.420 |
| 8 | 117.421 / 116.428 | 86.308 / 85.794 | 1348.569 / 1340.529 |

Depth-8 event speedup is 4.5583x / 4.5492x versus D1. Effective core rate
falls from ~1628 MHz at D1 to ~1475 MHz at D8 as matrix/VALU duty increases;
the user accepts the chip's known math-load clock response. Thus event
speedup need not equal the core-cycle ratio; no clock-gap tuning is attempted.

### FP16 payload control, U=8

| D | TFLOP/s, run 1 / run 2 | Useful weight GB/s, run 1 / run 2 |
|---:|---:|---:|
| 1 | 18.010 / 17.955 | 562.828 / 561.083 |
| 2 | 32.236 / 32.373 | 1007.363 / 1011.661 |
| 4 | 43.252 / 42.943 | 1351.620 / 1341.972 |
| 8 | 44.202 / 44.229 | 1381.324 / 1382.155 |

The larger 8192-step run gives FP8 D8 ~85.343 TFLOP/s, 1333.481 useful GB/s,
117.051 cycles/MFMA. FP16 D8 is ~44.591 TFLOP/s, 1393.458 useful GB/s.
U=32 FP8 D8 is ~85.958/~86.005 TFLOP/s in the two 4096-step runs, not a
uniform improvement over U=8. All raw data and per-sample ranges are retained.

All **420 timed samples** and **120 changed-input/poisoned-output/guard cases**
pass with zero absolute error versus FP64. Short 32/96-step controls also
exercise queue priming, wraparound and final drain; they are correctness
checks, not steady-state performance measurements.

## Scope and stopping point

The requested obvious queue-depth change gives a large, repeatable gain;
**D8 is best among these tested depths**. It is not yet the ~64-cycle
register arithmetic rate, and no claim of compute-bound general GEMM is made.

A availability is idealized exactly as the user requests. This experiment
changes neither operand reuse nor M/N tiling, and makes no activation-traffic
roofline argument. It also does not retune the conversion split/backedge.
No fixed backedge constant is fitted through memory variability. The next
algorithmic decision remains with the user; no deeper queue or reuse campaign
is silently added.

## Reproduce

```sh
bash research/fp8-w8a16/build-pipeline-probe.sh
python3 research/fp8-w8a16/audit_pipeline_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/pipeline/pipeline-probe 4096 9 1729
# Repeat: 4096 9 2718. Larger control: 8192 3 31415.
python3 research/fp8-w8a16/analyze_pipeline_probe.py
```

Stdout is CSV, stderr records footprints/validation. Retained flags, hashes,
audit, raw runs, medians and speedups are in `results/pipeline/`; full ISA,
loops, binaries and metadata are in `temp/fp8-w8a16/pipeline/`. The analyzer
checks completeness, exact correctness, all changed-input validations and
payload accounting before reporting speedups. Engine code/CMake and earlier
research kernels are unchanged; engine tests were not rerun for this isolated
extension. The committed register checkpoint remains `8e1c120`.
