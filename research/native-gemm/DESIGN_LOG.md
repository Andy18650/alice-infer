# Native GEMM implementation record

This is the early experiment record, not the current design summary. Start
with [DESIGN.md](DESIGN.md) for the final FP16/scaled-FP8 architecture, results
and lessons; follow its links for later scheduling/scaling experiments.

All studies below are native HIP O3/C++20 gfx90a HIP0/BDF31:00.0, CPU+memory
NUMA0, no model/context or clock overrides, packing/H2D/checking outside the
kernel interval. No library GEMM, including references. Preserve baseline
executables rather than overwrite; source/native SHA256 and actual ISA
accompany each phase. Screening pilots use three launches after200 ms warmup;
only the final table is two counterbalanced nine-launch process repeats.

## Initial K32 schedule

128/256 square CTA tiles, wave64/128 square register tiles, four waves,
one16/32 KiB LDS buffer and five raw K8 slots. Retire current LDS readers,
write prefetched next panel, fetch future panel, compute q0/q1/q2, publish
next LDS, batch all next operand reads, compute q3. Correct for K0/odd/tails,
ragged M/N and several stage wraps. No spills.

First104-CTA pilots:128 at1664x1024x8192 ~106.26 TFLOP/s,~71.6% local CU
body efficiency;256 at3328x2048x8192 ~117.75,~78.1%. These are working
baselines, not peak claims. Body efficiency differs from whole-kernel/event
and later multi-resident global efficiency.

Evidence: `results/initial/`; native executables `temp/native-gemm/t{128,256}/`.
Initial full/ragged short-check logs are retained in `temp/native-gemm/logs/`.

## Null hypotheses retained, not promoted

**VGPR pair-position control:** register-only native MFMA, four independent
accumulators, identical FP16 inputs/output, compare v[0:1] with v[2:3] versus
v[4:5]. Four samples each give~64.00 cycles/four-wave MFMA and every output
exact2048. The simple same-pair-position explanation is not supported by
this control. It does not prove all possible register banking irrelevant.
No operand-layout rewrite was made on this basis. Source/ISA/fingerprints and
raw results: `results/bank/`; actual program `temp/native-gemm/bank-probe`.

**LDS permutation:** matched runtime choice linear lane versus
`lane XOR (lane>>3)` for producer AND consumer LDS addresses only; global
packing and timed instruction counts unchanged.128 ~106.13/106.20 and256
~117.48/117.38 TFLOP/s for linear/XOR. No useful gain; default stays linear.
Evidence `results/swizzle/`, binaries `temp/native-gemm/t{128,256}-swizzle/`.
This tests one permutation, not all bank layouts.

## Arithmetic versus operand-pipeline control

`resident` build seeds raw operands from the first packed K32 stage once,
then repeats the normal q0/q1/q2/q3 MFMA/register sequence without inner-loop
loads/stores/barriers. CPU reference explicitly repeats that first fragment;
this is **not ordinary GEMM** and not relabeled as a full independent-matrix
result. All reference points/finite outputs/guards pass in the constrained
exact integer/16 domain.

At the104-CTA pilots, resident body efficiency reaches~98.0% for128 and
~99.3% for256, versus the real GEMM~72/78. Clock ratios are similar enough
that this is a real operand-pipeline gap, not merely a raw-rate comparison
across different clocks. Resident whole-event TFLOP/s141.97/146.44 is NOT
its body-only percentage times a fixed nominal clock; setup/output remain.
Controls in `results/resident/`, programs `temp/native-gemm/t{128,256}-resident/`.

## Earlier publication / rolling reads

Preserve retire-before-write barrier. Publish next LDS after current q0,
read next q0 and spare q3, then interleave next q1/q2 reads after their
current math and finish current q3. Same memory/math/barrier counts and
resource footprint; next reads get more matrix work to hide behind.

Matched pilots improve128 ~106.20 ->113.08 TFLOP/s, local body~71.6 ->77.2%;
256 ~117.46 ->122.75,~78.0 ->82.0%. Adopt this schedule. Evidence in
`results/late/` and `results/early/`; executables `temp/native-gemm/late-t*`
and `temp/native-gemm/final-t*`. Those historical names do not mean the
latest measured full-problem run; final suite is separately recorded below.

## K64 regression: stopped, kept K32

Increasing K32 to64 doubles source/operand staging, halves barriers per math,
expands live raw slots5->9, and raises LDS16/32 KiB ->32/64 KiB. Publish after
the first quarter of current math in both. Both compile with zero private
segment, reported184/480 registers for128/256, and pass fill/tail checks.

But104-CTA pilots regress to~105.05/~113.47 TFLOP/s, local body~71.0/~75.1%.
Keep K32 as default. No subsequent schedule/permutation/tiling optimization
was tried. Larger staging does not automatically improve overlap. This is
not a barrier-only experiment; live registers, LDS traffic bursts and
residency differ, so no unique regression mechanism is claimed.

Evidence `results/stage64/`, executables `temp/native-gemm/s64-t*`. Its original
source was recovered after known host-only output/default edits; recovered
probe/data/generator files **byte-match every recorded source SHA256**. It is
not silently described as a fresh later binary validation. Original generated
headers/native objects/flags remain unchanged. Reproduce with explicit StageK64.

## Full-grid comparison changes the choice

The128 candidate permits three CTAs/CU;256 only one. At larger output grids,
multiple128 CTAs hide much of each other's stalls. Raw steady-clock intervals
show maximum312 simultaneous CTA bodies for the representative long-K128
launch versus104 for256. This supports multi-resident latency hiding, not
an isolated occupancy-only causal proof: tiling/packing/reuse also differ.

Actual complete independent-matrix GEMM at4096x6656x32768 reaches144.07/
144.17 TFLOP/s for128/K32 versus128.37/128.35 for256/K32. Selected kernel
uses~1418 MHz effective core rate; whole-GCD clock-adjusted efficiency~95.5%.
Per-CTA share~32.6% is expected when three resident CTAs share a CU. Never
interpret it as32.6% whole-device utilization.

128 also wins the other three matched full problems, particularly the
1664x1024 case where256 produces only28 CTAs and padding versus104 for128.
No claim that256 is universally worse, or that this128 tile is globally
optimal across other data/types/shapes. Other constraints/input formats,
core/power states or end-to-end packing can change results.

Final evidence: `results/final/`, selected executable `temp/native-gemm/s32-t128/`.
The180-launch final suite includes functional seeds/tails and timed comparisons;
large outputs are sampled-reference plus all-C finite/guard checks, not full
CPU dot products. Event times, per-wave binary clocks, overlap proof and
normalization calculations are retained. No production changes.
