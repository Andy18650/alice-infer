# Packed FP32 moves and scale FMAs on gfx90a

**Correction/rerun:** the first trial (`results/scaled-packed/`, commitda408541)
had incorrect packed timestamp selectors and is marked INVALID for clock
metrics. Its initial expansion auditor also modeled that default incorrectly.
The final results here are from a fresh full rerun in `scaled-packed-fixed`,
with explicit selectors, a hardware raw-bit gate/negative control, and an
independent event-clock containment check. Original files are retained.

User-directed change to the K128/N128 scaled kernel. Two increments are
compared in one counterbalanced three-variant run: original scalar operations,
packed moves only, then packed moves+FMAs. No load/read/fence/bank/scale layout
or scheduling retuning. Fixed256 architectural VGPR/0 AGPR/51 SGPR,private0,
16 KiB LDS and runtime cap2 CTAs/CU in all variants.

## Exact replacements

Initialization of scaledv64:127 and dummy raw bankv192:255:

```text
v_mov_b32 v64,0; v_mov_b32 v65,0
    -> v_pk_mov_b32 v[64:65],0,0
```

128 scalar zero moves become64 packed moves. Timestamp low/high transfers
also use one packed move instead of two scalar moves:

```text
v_mov_b32 v2,s4; v_mov_b32 v3,s5
    -> v_pk_mov_b32 v[2:3],s[4:5],s[4:5] op_sel:[0,1]
```

Both sources are the same aligned pair, and explicit `op_sel:[0,1]` copies
its low and high words. V_PK_MOV defaults to selecting the LOW word for both
destinations; unlike packed FMA, it does not default to low/high selection.
This is a raw-bit copy, not a floating-point conversion. Four timestamp pairs remove four
more instructions. Unrelated scalar moves and single-register address setup
stay unchanged.

For each two consecutive scaled outputs, two scalar FP32 FMAs become one
`v_pk_fma_f32`. Its two vector operands select consecutive low/high values.
The scale is an aligned **SGPR pair**, but both halves must select the SAME
FP32 scale, not adjacent current and previous scales:

```text
current scale s14:
  v_pk_fma_f32 v[64:65],v[128:129],s[14:15],v[64:65]
      op_sel:[0,0,0] op_sel_hi:[1,0,1]
previous scale s15:
  v_pk_fma_f32 v[64:65],v[192:193],s[14:15],v[64:65]
      op_sel:[0,1,0] op_sel_hi:[1,1,1]
```

Gfx90a accepts that SGPR pair source. The disassembler omits default selector
fields. `s[15:16]` is rejected because of alignment, not because all scalar
sources are prohibited. No VGPR copy/duplication of the scale was added.

The eight-stage hot loop goes from128 scalar scale-FMA instructions to64
packed ones. Each16-instruction microtile drain becomes eight packed FMAs;
final64-instruction drain becomes32. There is no additional movement of the
remaining work into the freed issue window. MFMA issue ordering and bank
handoff/retirement gaps remain the same. Instruction count reduction does NOT
imply a corresponding doubling of full-kernel throughput or an assertion
about individual instruction latency.

## Validation

**568 checked launches/232 hashed processes**:190 correctness launches across
the two new variants and378 timing launches across all three. Existing exact
int64-partial/ordered-std::fma CPU oracle, non-power-of-two/zero/nonuniform
scales across K/N, K128/eight-stage tails, long K65536, input seeds and finite
FP8 cases all pass. Every output is scanned finite after NaN poison, guards
checked; large references retain the original sampling policy.

A dedicated GPU gate checks74048 exact64-bit copies (asymmetric words, every
bit position,complements,raw NaN/sign-bit patterns and random inputs). The
omitted-selector negative control reproducibly duplicates low32 and rejects
73920 unequal-word outputs. The corrected expansion auditor rejects the old
default-selector assumption. A regression rejects all28 affected old timing
processes/252 samples while accepting the14 scalar controls.

Every timed launch now also requires instrumented body span <=independent HIP
event duration (with a small tolerance), not merely consistent ratios computed
from the same dump. It validates raw per-wave timestamps, monotonicity, full
grid steady span, core MHz and useful throughput. Thus the
new timestamp pair moves are checked through the actual stored64-bit values,
not only source spelling.

Audit expands ONLY the exact emitted packed moves/FMAs into scalar operations:

- Full generated instruction flow matches the scalar baseline exactly.
- Full native kernel up to ENDPGM matches exactly after expansion and mapping
  branch targets into expanded instruction coordinates (raw byte offsets
  naturally change). This includes setup, tails, stores and timestamps.
- Pair alignment and both scale selectors are explicitly checked; all three
  variants retain identical resource metadata.
- The expanded flow passes the existing29-stage-count raw-bank/scale ownership,
  ordered group contribution and no-overfetch model for every variant. This
  is not a GPU latency emulator. Actual GPU references are a separate gate.

No original archive/source/native executable overwritten. The original scaled
comparison now pins its source snapshots because the live generator changed.

## Matched results

Native HIP O3/C++20 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0,block256,automatic
clocks,no model/context/clock override. Full useful GEMM FLOPs/full HIP event
time; packing,H2D,scale upload,reference checking excluded. Two nine-sample
process medians per variant/problem; reverse variant+problem order in repeat2.

K32768 TFLOP/s,run1/run2:

|CTAs|Scalar|Packed moves|Packed moves+FMA|Final efficiency|
|---:|---:|---:|---:|---:|
|104|118.757/118.898|118.939/118.797|120.271/120.292|76.79/76.78%|
|208|135.824/135.758|135.851/135.877|136.944/136.944|89.77/89.74%|
|312|137.941/137.868|137.959/137.922|139.104/139.012|91.59/91.50%|
|1664|142.481/142.548|142.512/142.504|143.433/143.431|94.65/94.66%|

M/N respectively1664/1024,3328/1024,4992/1024,4096/6656. K8192 tests also
retain104/208/312; all cases in `results/scaled-packed-fixed/comparison.md`.

Packed moves alone are within measured timing noise. Packed FMAs provide a
modest consistent long-K improvement (~0.6-1.3% relative to scalar); the
removed operations were only part of the kernel and shared existing matrix
execution windows. The run is not an isolated packed-FMA latency/throughput
measurement, nor a zero-stall proof. No scale-prefetch or further interleaving
optimization was tried.

## Build/replay and records

```sh
bash research/native-gemm/build_scaled.sh scaled <fresh-mov> mov
bash research/native-gemm/build_scaled.sh scaled <fresh-packed> all
python3 research/native-gemm/run_scaled_packed.py --mov-build <fresh-mov> --packed-build <fresh-packed> --output <fresh-results>
python3 research/native-gemm/audit_scaled_packed.py --output <fresh-audit>
python3 research/native-gemm/verify_artifacts.py
```

Default new builds use `temp/native-gemm/scaled-packed-fixed/scaled-{mov,all}`;
`build_scaled.sh` defaults to `all`. Direct generator defaults to `none` for
explicit historical regeneration. Old scalar builds remain under `scaled/`.
Matched sources/native/resource/raw clocks/commands/actual driver records:
`results/scaled-packed-fixed/`; expansion/scale-selection/order proof:
`results/scaled-packed-fixed-audit/`. The previous trial/proofs are explicitly
withdrawn, not rewritten. Raw `.bin` dumps are marked binary in git attributes;
that classification does not modify any captured bytes.
No production/CMake/model/session/system/package/clock changes. Research-log
working copy appended, unrelated user changes remain unstaged.
