# Native gfx90a FP16 GEMM: selected128x128x32

**Start with [DESIGN.md](DESIGN.md)** for the concise FP16/scaled-FP8 design,
final results and development lessons. This README retains detailed historical
comparisons and build controls; [AUDIT.md](AUDIT.md) covers code/artifact review.
The follow-up [practical-shaped study](../practical-gemm/README.md) compares
ordinary row-major A with this packed-A baseline, including GPU packing cost;
its first loader is correct but not promoted over the packed kernel.

From-scratch full-grid GEMM, **no rocBLAS/CK or other GEMM library linked or
called**, including validation. HIP runtime plus native global/LDS/MFMA ISA.
FP16 A/B operands, FP32 accumulators and row-major C, alpha1/beta0, no split-K
or atomics. Standalone research, not engine integration.

**K128/N128 FP32 scales implemented:** [SCALED_FP8.md](SCALED_FP8.md).
All192 accumulation registers plus the full64-register operand/producer state
fit in256 architected VGPRs,AGPR0/private0,two CTAs/CU. Scalar-address/lifetime
compaction has no observed performance loss; separate controls quantify the
occupancy and scaling costs. [Packed FP32 moves/FMAs](PACKED_FP32.md) retain
that budget and improve long-K useful rate by~0.6-1.3%; packed moves alone
are within noise. The original packed timestamp trial is withdrawn; the linked
report uses the corrected-selector full rerun and independent bit/clock gates. The unscaled defaults remain supported.
[Scale-latency experiments](SCALE_LATENCY.md) implement all three next proposals:
late scale loads give~125.4/137.3/139.6 TFLOP/s at104/208/312,K32768 and~144.6
on the large grid. Pair batching/earlier q1 reads are mixed and not promoted.
[Balanced q0 scheduling](Q0_SCHEDULE.md) then moves future A loads early and
spreads q0 work into13/15/15 native-instruction gaps:~127.9/139.0/141.4 TFLOP/s
at104/208/312,K32768 and~146.2 large. Early A loads alone are slightly negative.

**Separate FP8-B experiment:** [FP8_B.md](FP8_B.md) distributes A/B K8
production across all four waves (two A x4 loads + one FP8 B x4 each), expands
B before LDS, and passes exhaustive finite conversion/whole-GEMM checks. The
first conversion-burst schedule regresses104/208/312 grids (~120.6/~134.0/
~139.0 TFLOP/s atK32768); large grid remains~146.5. It is retained under
`fp8-b`, **not promoted over the FP16 default**. Packing changes are explicit.
[Later FP8 scheduling](FP8_SCHEDULE.md) clusters copies/prefetches B early and
interleaves conversion with current math: ~124.6/136.7/140.9 TFLOP/s at104/208/
312 CTAs,K32768,~147.35 on the large grid. Two matched increments are preserved.

**For process/code review, start with [AUDIT.md](AUDIT.md).** The active code
has been expanded into named kernel, packing, reference, timing and validation
steps. Original source/results remain frozen. The readability-only cleanup
had byte-identical generated headers/GPU code; that historical proof is pinned
to `results/readable/source/`. The later intentional [branch-only change](BRANCH_CLEANUP.md)
uses a different control-flow audit; it does not claim unchanged whole GPU code.

**Selected design:**128x128 output tile, four waves/CTA, K32 pipeline,16 KiB
single LDS buffer, wide global/register staging and b128 LDS transfers.
After branch cleanup and [one MFMA moved after head VM0](MFMA_FIRST.md),
the largest check reaches **146.55/146.44 TFLOP/s (~97.1%)**.208-CTA long-K
reaches **137.90/137.99 TFLOP/s (~90.6%)**. The future global packets remain
batched; no broader prefetch interleaving change has been made.
This is near-compute-bound in these conditions, not proof of a global optimum
or a general GEMM/prefill performance guarantee.

## Original matched tile comparison (before branch cleanup)

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, no model/attention
context or clock/power overrides. Two counterbalanced process repeats, nine
checked eager launches each;200 ms selected-kernel warmup. CPU checking
between launches is outside timing; these are not queued graph batches. Useful FLOPs divided by
**full kernel HIP-event time**, including accumulator initialization/output.
Packing, H2D, CPU reference and checking are excluded. The largest problem
has672 MiB of distinct packed A/B operands and104 MiB of C: this is not a
whole-input L2-resident or repeated-fragment GEMM. Cache reuse is between
CTAs' current K panels, not a claim that the entire input remains cached.

| M x N x K | Selected128 TFLOP/s | Candidate256 TFLOP/s | Selected clock-adjusted whole-GCD efficiency |
|---|---:|---:|---:|
|1664x1024x8192 |113.15 /113.23 |34.46 /34.42 |72.1% /72.0% |
|3328x2048x8192 |137.66 /137.66 |122.66 /122.68 |90.6% /90.5% |
|4096x6656x8192 |141.58 /142.07 |124.59 /124.92 |93.5% /93.7% |
|4096x6656x32768 |**144.07 /144.17** |128.37 /128.35 |**95.4% /95.5%** |

Selected effective core/steady-clock ratios are~1475,1428,1424,1418 MHz for
these shapes, not nominal1700 MHz. Global efficiency estimate is
`useful_TFLOPs/(104*1024*median_core_MHz/1e6)`; it uses the measured ratio,
not a fixed raw-register benchmark clock. It is an operating-point estimate,
not a clock-locked hardware peak or exact frequency-weighted integral.
Raw event times, clocks and reference counts are in `results/final/`.

## Why128, not the larger tile

A four-wave128 tile has a64x64 register tile/wave: four independent32x32
accumulator tiles,64 AGPR values/lane. Its **requested global input intensity
is64 FLOP/B**, twice the previous64x64 CTA scheme, and LDS-operand-read
intensity is32 FLOP/B, also doubled by within-wave A/B reuse.

Per K32 stage:16 KiB global reads,16 KiB LDS writes,32 KiB LDS reads,
1,048,576 FLOPs/CTA. Four waves cooperatively stage two64-row A panels and
two64-column B panels; all waves compute. Each loaded A/B microfragment is
used twice within its wave, and each LDS panel is shared by two waves.

HSA reports136 VGPRs,64 AGPRs and46 SGPRs; **do not blindly sum these metadata
fields as independent physical pools**. LDS is16 KiB, private segment zero.
HIP occupancy query allows three256-thread CTAs/CU. The representative large
launch's raw steady clocks show **312 overlapping CTA-body intervals**, versus
104 for the256 candidate. This is overlap evidence, not exact allocation
timestamps or a CTA-to-CU map.

The256 tile has higher intensity but256 accumulators/lane and reported384
registers, permitting only one CTA/CU. Its local pipeline gaps remain exposed.
The multi-resident result is consistent with hiding those gaps; this is not
an occupancy-only causal isolation, because tiling/packing/reuse also differ.
A per-CTA share of
~32.6% of a CU's peak is compatible with~95.5% whole-GCD efficiency when
three CTAs share the CU; it is not evidence that the GPU is~67% idle.

For1664x1024,128 gives104 CTAs whereas256 gives only28 and some padding:
the larger tile severely underfills the device.128 wins every tested full
problem, so it is the default.256 remains an explicit research control, not
an auto-dispatched path or a recommendation for untested shapes.

A follow-up [104/208/312-CTA coverage study](GRID_COVERAGE.md) finds about
128 TFLOP/s at208 CTAs/K8192 and133 atK32768. Grid size is not an enforced
per-CU residency cap; that note separates observed overlap from placement.

## Pipeline / correctness of buffer reuse

One opaque asm owns accumulators, operand registers and producer state over
all K. Four K8 slices/stage, plus one extra raw slot for the last slice:

1. Startup loads stage0, writes/publishes LDS, reads its four operand slices
   into registers, and prefetches stage1 globally into a separate register set.
2. Steady head waits for all current operand reads, then a CTA **retirement
   barrier before LDS overwrite**. Wait for prefetched global input.
   Immediately issue current q0/C00 MFMA, then write next stage to the same
   LDS buffer and issue fresh global loads for stage+2.
3. Issue the remaining current q0 MFMAs, wait for next LDS writes and **publish with a barrier**.
   Read next q0 into consumed q0 registers and next q3 into the spare slot.
4. Compute current q1/read next q1, compute current q2/read next q2, compute
   current q3. Those reads overlap independent matrix work.
5. Two-phase unrolling swaps which fixed register slot holds q3, without a
   VGPR copy. Repeat. Tail consumes the last stage without fetching beyond K.

Two barriers per K32 stage, rather than one per K8 MFMA. Two separate purposes:
retirement before writes, publication after completion. `s_barrier` is not
assumed to drain memory. Next raw reads never precede publication. Current
operands are in registers before overwrite; all waves join before reuse.

Native hot backedge (two K32 stages)/wave:32 MFMA,8 global DWORDx4,8 LDS b128
writes,16 LDS b128 reads, four barriers, two VM0 waits, four LGKM0 waits.
The current control flow has only subtract-by-two,compare,conditional backedge:
81 instructions instead of90. The first current-q0 MFMA is now immediately
after head VM0, before staging stores; counts/resources stay unchanged.
The final1–3 stages are peeled outside the steady loop.
[BRANCH_CLEANUP.md](BRANCH_CLEANUP.md) gives parity/bounds proof and matched
104/208/312-CTA data for that historical control-only step. [MFMA_FIRST.md](MFMA_FIRST.md)
documents the subsequent intentional single-MFMA reorder.
No datatype conversion, accumulator copies, lane moves, scratch spills or
NOPs in that loop. VM0 operates on the prefetched panel after the preceding
stage's matrix overlap; it is not a per-global-load serialization. A constant
conservative matrix-completion NOP gap occurs only at kernel tail, before
end clocks/AGPR harvest, not per stage.

## Input format / scope

Kernel consumes **prepacked FP16 panels**, not arbitrary row-major pointers.
The harness constructs full independent original logical matrices from the
nine signed values[-4..4]/16, bit-packs into real FP16 operands, and uploads
before measurement. Patterns vary across all logical coordinates, rather than
repeating one K fragment. Other operand distributions can change clocks/rates. It is not a procedural/no-load MFMA kernel. FP32 C is row-major.

Physical A layout:
`[K32 chunk][M64 panel][K8 slice][micro-pair][lane64][4 dwords]`.
B uses corresponding N64 panels, derived from original B[K,N]. Each16-byte
lane vector contains two32x8 MFMA fragments (four halves each), so one b128
LDS read supplies two microfragments. K-major panel storage encourages
cross-CTA row/column reuse. Host packing is recorded in `pack_ms`, not hidden
in reported GEMM time. It is not an optimized production packer. Real engine
use still needs device activation conversion/packing and load-time weight
packing; no real-model prefill or end-to-end row-major claim is made here.

M/N tails pad packed inputs to the output tile; K pads to32. Output stores
are masked to actual M,N, including K0 => zero C. Useful and padded executed
TFLOP/s are both recorded. CLI bounds M,N<=16384,K<=65536 are harness limits,
not claimed ISA limits. No general alpha/beta/transposition/batched API.

## Verification and controls

Original final/cleanup suite: **180 verified launches** =36 functional/input-
seed cases plus144 timed launches. The later branch and one-MFMA A/B experiments each have **312 checked
launches** (60 bounded-tail/seed checks +252 timings), plus10 optionalStageK64
boundary checks each. These are separate datasets, not relabeled old results. Small cases use full original-row-major CPU references;
large cases check~2–3K sampled/boundary outputs, scan **all C finite after
NaN poisoning**, and check32 guards every launch. Do not relabel large sampled
references as full references. Exact FP32 equality is justified for this
integer/16 domain and K<=65536, not all IEEE FP16 inputs. No library reference.

[DESIGN_LOG.md](DESIGN_LOG.md) preserves the initial schedule, null VGPR-pair
and LDS-XOR controls, correctly verified resident-fragment diagnostic,
earlier publication improvement and **K64 regression**. K64 is not default.
No losing variant silently replaced a baseline binary. Selected source and
native manifests, instruction bodies, per-wave binary clocks, samples,
checks, dynamic dependency listings and overlap proof are committed under
`results/`; actual measured executables/objects remain in `temp/native-gemm/`.

## Build / replay

```sh
bash research/native-gemm/build.sh                      #128,K32 ->temp/native-gemm/fp16-latency-host-bridge/s32-t128
numactl --cpunodebind=0 --membind=0 temp/native-gemm/fp16-latency-host-bridge/s32-t128/probe \
  --m 4096 --n 6656 --k 32768 --rounds 9 --warmup-ms 200
# Historical one-MFMA A/B uses preserved builds/source snapshots:
python3 research/native-gemm/run_mfma_first.py --output <fresh-A-B-directory>
# Optional expanded FP16 comparison also retains208/312 coverage:
bash research/native-gemm/build.sh 256
python3 research/native-gemm/run.py --output <fresh-results-directory>
python3 research/native-gemm/analyze.py <results-directory>
# To inspect old data without overwriting its derived reports:
python3 research/native-gemm/analyze.py research/native-gemm/results/final \
  --report-dir <fresh-audit-report-directory>
# Separate experimental FP8 B (128x128x32 only):
bash research/native-gemm/build_fp8.sh
# Historical A/B reuses preserved schedule builds/source snapshots:
python3 research/native-gemm/run_fp8_schedule.py --step interleaved --output <fresh-FP8-A-B-directory>
# Historical packed comparison (use archived sources for historical rebuilds):
bash research/native-gemm/results/scaled-packed-fixed/mov/source/build_scaled.sh scaled <fresh-mov> mov
bash research/native-gemm/results/scaled-packed-fixed/packed/source/build_scaled.sh scaled <fresh-packed> all
python3 research/native-gemm/run_scaled_packed.py --mov-build <fresh-mov> --packed-build <fresh-packed> --output <fresh-scaled-results>
# Current balanced-q0 experiment; staged comparison in Q0_SCHEDULE.md:
bash research/native-gemm/build_scaled.sh scaled <fresh-balanced> all balanced
# Historical register-budget comparison uses preserved scalar builds:
python3 research/native-gemm/run_scaled.py --output <fresh-budget-results>
```

`build.sh [128|256] [output-directory] [gemm|resident] [32|64]` supports only
the studied candidates/explicit diagnostics. Default is selected128/K32,
linear LDS lane mapping; `--swizzle 1` is opt-in null control. `resident` is a
repeated-first-fragment diagnostic, **not ordinary GEMM**; it is never used
by the final suite. Stage64 and archived CLI/header versions belong to their
recorded paths/snapshots. Use fresh directories to preserve measured binaries.
Normal builds now default under `temp/native-gemm/fp16-latency-host-bridge/`
and unscaled FP8 under `fp8-latency-host-bridge/` to preserve earlier measured
binaries after the shared host additions. Both original FP16 tiles and the
unscaled FP8 whole GPU .text remain byte-identical. Scaled modes use `scaled/`;
all older measured namespaces remain untouched. CSV column names and timestamp
binary layout are unchanged. The expanded default full runner has360 launches;
the focused branch A/B and original180-launch suite remain distinct.

No production/CMake/model-format/session API, package/system or clock-policy
changes. Status appended to `research/research-log.md`, not engine-design.md.
