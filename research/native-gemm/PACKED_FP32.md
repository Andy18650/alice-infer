# Packed FP32 moves and scale FMAs on gfx90a

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
    -> v_pk_mov_b32 v[2:3],s[4:5],s[4:5]
```

Both sources are the same aligned pair: default selectors copy its low and
high words, not a floating-point conversion. Four timestamp pairs remove four
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

Every timed launch also validates raw per-wave timestamp dumps, including
monotonicity, full grid steady span, core MHz and useful throughput. Thus the
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
|104|118.817/119.040|118.838/119.101|120.002/120.147|76.60/76.73%|
|208|135.798/135.798|135.811/135.824|136.944/136.809|89.74/89.69%|
|312|137.941/138.041|138.032/137.968|139.114/139.086|91.59/91.56%|
|1664|142.594/142.554|142.563/142.574|143.490/143.517|94.72/94.73%|

M/N respectively1664/1024,3328/1024,4992/1024,4096/6656. K8192 tests also
retain104/208/312; all cases in `results/scaled-packed/comparison.md`.

Packed moves alone are within measured timing noise. Packed FMAs provide a
modest consistent long-K improvement (~0.7-1.0% relative to scalar); the
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

Default new builds use `temp/native-gemm/scaled-packed/scaled-{mov,all}`;
`build_scaled.sh` defaults to `all`. Direct generator defaults to `none` for
explicit historical regeneration. Old scalar builds remain under `scaled/`.
Matched sources/native/resource/raw clocks/commands/actual driver records:
`results/scaled-packed/`; expansion/scale-selection/order proof:
`results/scaled-packed-audit/` and expanded resource check
`results/scaled-packed-audit-final/`.
No production/CMake/model/session/system/package/clock changes. Research-log
working copy appended, unrelated user changes remain unstaged.
