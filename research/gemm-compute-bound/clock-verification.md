# Review of clock-verification tests

Reviewed without modifying user source/logs. Located three newer tests:
`int8_clock.hip`, `mixed_clock.hip`, `gemv_clock.hip`, plus their prerequisite
`operand_clock.hip` (BF16/FP16 baseline), all under research/gemm-compute-bound.
Original results/firmware monitor data are under temp/raw-mfma.

Copies, fresh native -O3 C++20 gfx90a builds/runs, extracted code objects and
assembly are preserved under `temp/clock-review/`. Curated original/fresh logs,
source/binary hashes and phase-aligned firmware medians are committed under
`results/clock-verification*` and `results/*_clock-{original,fresh}.log`. Fresh runs used HIP_VISIBLE_DEVICES=0 and
numactl --cpunodebind=0 --membind=0. No clock/power/system settings changed.

## Overall verdict

Clock-ratio and MFMA operation-count arithmetic are sound. Original and fresh
clock results agree closely (the bus-limited bandwidth cases vary by a few percent); forward/reverse phases repeat the qualitative effect.
Firmware average-GFX data corroborate the ordering while current-GFX remains
1700 MHz. No private-segment allocations occur in the reviewed original binary
kernels. This validates workload/data-dependent effective-rate observations,
not a specific clock/power mechanism or every roofline/general-model inference.

## ISA/count checks

- BF16/FP16 operand baseline: four correct native 32x32x8 instructions per
  non-unrolled runtime iteration, no hot-loop global loads.
- INT8: four native 32x32x8i8 or 16x16x16i8 instructions. The counts use 2 ops
  per MAC, all waves/CTAs/chains/iterations, consistently with the float tests.
  uint32 output reduction/host expected values implement modular wrap safely.
  Results characterize these opcodes, not an untested double-K INT8 peak.
- Mixed: 32 MFMA instructions per eight-substep unrolled hot-loop body, 512
  loop iterations. Actual scalar/dwordx2/dwordx4 VMEM widths match 4/8/16 B
  per thread. Eight prologue plus 4096 replacement loads are counted; final
  slots are consumed, and the emitted loads remain live. 512 MiB buffer and
  power-of-two element masks are in bounds/aligned.
- GEMV: four scalar weight dword loads and two activation dwordx4 loads in
  the loop, SIMT FMA/VALU, no MFMA. Matrix/chunk/stamp/partial extents are safe
  for all listed phases. qbf16 truncation matches the historical test path.

## Supported findings

- BF16/FP16 constant tiny and constant quarter: ~1572 MHz, ~166.7 TFLOP/s.
  BF16 random ~1427 MHz /151.2 TFLOP/s; FP16 random ~1438 /152.4. Core-cycle
  work stays ~255 FLOPs/cycle. Magnitude alone does not explain the difference.
- INT8 random also reduces effective rate (~1435 MHz large /1474 small versus
  ~1570-1603 constant); the effect is not exclusive to floating-point formats.
- Adding independent streaming reads changes clock and issue efficiency.
  8 B constant mode: ~1511 MHz, 153 TFLOP/s, 1.20 TB/s requested bytes;
  8 B random mode: ~1389 MHz, 140.6 TFLOP/s, 1.10 TB/s. This resembles the
  previous GEMM effective rate, but is not the same dataflow as actual GEMM.
- SIMT GEMV gup/down reaches ~1.27/1.30 TB/s of useful weights at ~1550-1556
  MHz. High bandwidth alone does not universally imply the ~1385 MHz MFMA
  GEMM state.

## Important qualifications / review findings

1. GEMV times PRODUCER ONLY. k_combine runs in verify() but is absent from the
   timed launch lambda. Do not compare this as end-to-end GEMV against prior
   producer+combine measurements. Source lines 201-244 /290-293.
2. The original GEMV log header says GFLOP/s but its ~1.3 values are TFLOP/s.
   Current source uses the correct TFLOP/s label/formula. Source and current
   binary postdate that log, so exact provenance of the old log cannot be
   reconstructed; a fresh source build reproduces the numeric rates and
   prints the corrected units.
3. Mixed-load goal >=250 FLOPs/core-cycle is NOT reached with loads: 4/8 B
   cases are ~243-244 versus ~255 pure; 16 B is ~113-121 and bus-limited.
   Do not call those phases pure clock-only losses or fully loaded matrix
   execution. Pipeline depth 8 does not itself prove every wait is hidden.
4. Mixed random helper differs from the operand baseline and there is no
   random-load0 phase in the mixed phase list, despite dispatch support.
   Comparing random-only in another file to random+memory is not a perfectly
   matched factorial comparison. Add that control if isolating the load cost.
5. GEMV correctness checks only y[0], before timing; all weight rows repeat,
   x is small nonnegative integers and exactly BF16-representable. This is a
   useful historical clock/bandwidth probe, not arbitrary real-weight/data
   correctness or a full-FP32-activation contract test. Final/all-row checking,
   random inputs and an uninstrumented control would strengthen it. Its
   zero-weight phase changes BOTH memory values and VALU operands, not one
   isolated power factor.
6. Mixed checksum checks determinism and >50% nonzero, but not actual variation
   across threads or an independently computed expected checksum. A constant
   nonzero vector would pass. ISA/source addressing checks establish live
   loads here; the advertised check alone is weaker than its comment.
7. FP16 random values use only 6 mantissa jitter bits (~0.25..0.2654), BF16
   uses 7 (~0.25..0.4980); distributions are not numerically matched. A and B
   use the same per-lane bit vector. Avoid quantitative dtype power claims.
8. Working-set size and requested load bytes do not measure physical HBM
   traffic or establish a universal saturated bandwidth. Historical 1290
   GB/s comments are not a contemporaneous hardware ceiling. Current/average
   clocks and throttle bits cannot by themselves identify a power policy.

Fresh build/run warnings are only two unused GEMV class fields. No user files
were edited. Source/binary hashes and per-phase monitor correlations can be
recovered from the artifacts here; original monitors were aligned to PHASE
unix timestamps after excluding the first 0.5 s of each phase.

## Reproduce the standalone probes

```sh
mkdir -p temp/clock-reproduce
for name in operand_clock int8_clock mixed_clock gemv_clock raw_mfma; do
  hipcc -O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra \
    research/gemm-compute-bound/$name.hip -o temp/clock-reproduce/$name
  HIP_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
    temp/clock-reproduce/$name > temp/clock-reproduce/$name.log 2>&1
done
```

`raw_mfma.hip` is an additional uninstrumented constant-operand reference.
Its introductory comments mislabel offered waves: 104/208/416 CTAs of 256
threads give 4/8/16 offered waves per CU on 104 CUs, not 1/2/4. These are
offered counts, not a measured residency guarantee. It validates one launch
before each timed configuration, not every timed launch; its operation-count
formula includes all four waves per CTA correctly. Its source is retained
without treating those comments as measurement evidence.
