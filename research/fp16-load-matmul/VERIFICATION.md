# Full-output verification of the FP16 model

User explicitly re-enables correctness testing after the timing-only global/
LDS studies. `verification.hpp` adds an exact host reference; verification is
now ON by default, with no change to the timed load/MFMA schedule for the
default two-ready-bank implementation. Original binaries are preserved.

## Result

**531 waited cases passed exactly**, all 4096 outputs and 32 guard words per
case. This includes **96 direct launches of the original saved code objects**,
not only recompilations. Cases cover:

- Global hot/L2-sized/streaming and LDS hot/sweep, all three load shapes.
- LDS depths 1/2/3/4/8, packed and deliberate bank-stride4 layouts.
- The original +/-0.125 data and record/lane/half-varying hash data, baseline,
  independently changed A, independently changed B, and both changed.
- D16/U1 and D3/U8, including short partial-phase exits (17/40 steps) and
  odd seven-record cyclic windows.

All waited errors and guard errors are zero. **8/12 no-wait characterization
cases fail numerically**; four observed matches are not safety guarantees.
The unsafe paths are never counted as certified correct.

A further **324 verified timing passes** also return exact outputs and intact
guards. The original device code is used for the historical-pattern timings;
new hash LDS tests use the rebuilt initializer. Fresh values are somewhat
higher than historical timing-only runs; no bit-identical timing claim:

| Original kernel | Verified recheck core cycles/step, run1 / run2 |
|---|---:|
| Global hot D8, 1xdwordx4 | 65.029 / 65.297 |
| Global hot D8, 2xdwordx2 | 129.078 / 129.077 |
| Global hot D8, 4xdword | 98.635 / 98.787 |
| LDS hot D2, 1xb128 | 65.408 / 65.237 |
| LDS hot D2, 2xb64 | 65.724 / 65.750 |
| LDS hot D2, 4xb32 | 66.791 / 66.814 |

Native HIP O3/C++20, gfx90a, HIP0 / PCI31:00.0, CPU+memory NUMA0, one
256-thread CTA/four waves, K32768, U32, no model/context or clock settings
changed. Checking is outside core/event timing but changes the host launch/
readback environment. The small timing shift is not uniquely attributed.

## Reference and inputs

The CPU reference independently reconstructs A[32][8] and B[8][32] from each
selected physical layout and computes integer dot products. Each constructed
FP16 operand is an integer /16, products integer /256. At steps<=8192 every
possible intermediate integer magnitude is <=2^20, well within FP32 exact
precision. Aggregating per-record occurrence counts is therefore equivalent
to the sequential device sum without rounding-order ambiguity. Comparison is
**exact numerical equality**, not a tolerance or checksum; signed-zero bits
are not separately compared.

The MFMA output register layout is explicitly mapped to row/column on CPU:
`row=8*(acc/4)+4*(lane/32)+acc%4`, `col=lane%32`. Input reconstruction uses
both lane halves and all four A/B halves. Every output is checked.

- `--pattern legacy` preserves the original physical-word sign pattern.
- `--pattern hash` varies by physical word/half with magnitudes 0..4/16 and
  independent `--seed-a`/`--seed-b` values. It is not period-two per-slot data.
- Global initialization is fully read back and compared with constructed
  host input before reference calculation. LDS is directly initialized using
  the same input construction, while reference addressing/math is separate.
- Outputs are poisoned, surrounded by 16-word leading/trailing guards. Host
  negative controls inject an incorrect value and each guard corruption to
  ensure the checker rejects them.

These tests do not cover arbitrary FP16 NaNs/infinities/subnormal flushing,
large inexact reductions, or production GEMM shapes. They target indexing,
load/readiness, operand lifetimes and the current dummy-data numerical model.

## Original-code-object checking

`--module OLD.hsaco --symbol OLD_KERNEL` uses `hipModuleLaunchKernel` with
the saved five-argument ABI. Host verification then checks the actual original
kernel. Global old kernels can consume new hash data in the external buffer;
old LDS kernels contain their legacy initializer and are tested with legacy
inputs only. Module launches require a two-ready-bank host build and matching
D/U/stride. Code-object SHA256s and exact symbols are recorded per case.

## Interface and artifacts

`--verify 1` is default; `--verify 0` explicitly opts out. Verification supports
steps<=8192. A mathematical/guard failure exits **2**, execution/argument errors
exit **1**. CSV includes `verified,mismatch_count,guard_errors,max_abs_error,
valid` plus ready-bank count. No-wait always remains semantically unsafe,
even if a particular dataset passes. Matrix runner continues past expected
unsafe mismatches but stops unexpected waited failures.

`verify_cases.py` writes `results/verification/`; `verified_timing.py` writes
`results/verified-timing/`. Both preserve complete commands, logs, CSVs,
module fingerprints and process snapshots. `--resume` on the verifier resumes
only an unchanged experiment; do not reuse cached cases after modifying code.

These scripts target the preserved pre-single-ready verified paths
`temp/fp16-load-matmul/dD-uU-ldsS-verified/`. The current build script can
reproduce those names via its explicit output-directory argument (R defaults
2). Original performance-only sources are additionally archived at
`temp/fp16-load-matmul/pre-verification/source/`. No engine/CMake changes.
