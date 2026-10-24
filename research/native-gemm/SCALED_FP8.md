# FP32 scales per K128 x N128 FP8 weight block

**Implemented successfully:** the full three-bank accumulation fits **256
architected VGPRs, zero AGPRs, zero private bytes**, and HIP reports **two
CTAs/CU**. No reduction of the five raw operand slots, StageK32, producer
payload, or conversion/prefetch overlap was necessary.

The register compaction itself has no observed performance loss. A compact
unscaled control is slightly faster than the previous144-register kernel.
The final scaled kernel has the expected additional cost, measured separately
from compaction and register-limited occupancy below. This is standalone
research, not engine/model-format integration.

## Mathematical and numerical contract

B stores finite E4M3FN bytes and a separate FP32 table:

```text
scales[ceil(K/128)][ceil(N/128)]          // K-group major, N-tile minor
B_real[k,n] = decode(B[k,n]) * scales[k/128,n/128]
P_g = sum over k in K128 group g of A[:,k] * decode(B[k,:])
C = ordered FP32 FMA(P_g, scale_g, C)
```

The N128-aligned CTA shares one scale across its four waves at a given K128
group. Each wave uses an SGPR scale, not64 replicated scale values. Four K32
stages form one raw partial. The final partial group uses zero-padded operands,
not a scale from the next group. M tiles reuse the B scale table.

For default A/B units[-4..4]/16, each raw128-term partial is exactly FP32.
The CPU oracle independently sums original logical operands in int64, then
uses `std::fma(float_partial,float_scale,float_accumulator)` in K-group order.
This tests the specified grouped FP32 rounding, NOT equality to an arbitrarily
ordered FP64/full-dequantized GEMM. Tests include non-power-of-two FP32 scales.
Finite-code FP8 tests retain signed A=1/16,K<=64 and exact partials. No new
all-IEEE/all-scale-range correctness claim (NaN/FNUZ remain excluded).

## Exact register budget

The old144 allocation includes64 AGPRs and80 allocated architectural VGPRs
(the code uses throughv76; the allocation is rounded). Simply adding two64-
register banks would give272 allocated registers and would not fit all-arch.

The new opaque pipeline passes **only SGPR inputs** and performs output/stamp
stores inside asm, so compiler-owned vector arguments/work-item IDs/output
arrays need not remain live across it. No scratch spill or hidden AGPR copy.

|Registers|Use|
|---|---|
|v0:39|The same five8-register current/rolling MFMA operand slots|
|v40|Shared lane byte offset `16*lane` for both global sources|
|v41|Persistent A LDS consumer address|
|v42|Persistent B LDS consumer address|
|v43|A producer LDS address; B uses immediate+8192|
|v44:51|A producer payload|
|v52:59|Expanded B payload|
|v60:63|Packed B raw buffer, early reload after all permutations|
|v64:127|Persistent scaled FP32 accumulation,64 values/lane|
|v128:191|Raw MFMA bank0,64 values/lane|
|v192:255|Raw MFMA bank1,64 values/lane|

Global source bases live in SGPR pairs and GLOBAL instructions use a32-bit
vector lane offset plus64-bit scalar base. Stage increments become scalar
add/carry operations. Four persistent vector addresses replace the previous
vector pointer pairs plus externally bound vector address inputs. B packing,
per-wave K8 ownership and all raw LDS offsets remain identical.

Metadata: **256 VGPR /0 AGPR /51 SGPR,private0,16 KiB LDS**. The scalar budget
includes current/previous scales, scale pointer/stride and timestamps. Output
full-wave/tail predicates are computed in SALU, avoiding a compiler-generated
vector boolean live across the all-VGPR asm boundary. Dynamic
LDS remains16 KiB and actual reported occupancy is2 CTAs/CU.

## Bank and scale pipeline

- Raw banks alternate every K128 group. The first K8 MFMA of each output
  microtile uses src2=0 to initialize the new bank, avoiding64 clear MOVs/group.
- While the new group's **q1 MFMAs** write the new raw bank,16 direct VALU
  FMAs per microtile drain the previous raw bank into scaledv64:127.
  q0 retains the existing early-B/copy/conversion interleave schedule.
- Previous/current scales stay in separate SGPRs. The scale table is read by
  SMEM once/wave/K128. The new scale load precedes the existing head LGKM0;
  that wait establishes availability. It is not a matrix-completion fence.
- The previous bank is consumed after the preceding group's final q3 and the
  new group's q0 work, in microtile completion order. The order audit tracks
  at least eight subsequent MFMA issues before these previous-bank reads;
  this is not an OOO-MFMA assumption or a latency simulation.
- A zero-initialized unused raw bank permits a harmless first-group dummy
  drain, avoiding an inner-loop first-group branch. One extra64-FMA zero
  drain occurs per nonempty kernel; no scaling table is read for K=0.
- After the final raw partial, a conservative64-cycle matrix-tail NOP gap
  precedes its64 scaling FMAs. Output clocks follow the final scale update.
- Eight K32 stages (two raw-bank groups) form the full steady unroll, restoring
  raw and operand-slot parity. Only bounded tails check future-load validity;
  no weight/scale overfetch. Startup, partial groups and K=0 are explicit.

The existing five-slot operand pipeline, retirement/publication barriers,
32-bit conversion arithmetic and one B global x4 per wave/stage remain.
Output stores have a uniform full-wave fast path and a masked M/N-tail path;
dead operand VGPRs serve as epilogue temporaries. Both are included in full
HIP event timing. This is a budget/address/epilogue/control port, not a claim
that only one scheduling instruction changed relative to the old kernel.

## Controls and matched performance

Four variants, same references/clocks/eager harness:

1. **Original:** archived FP8-interleaved144 allocation,64 AGPR,cap3.
2. **Compact:** the new scalar-address pipeline, one64-register architectural
   accumulation bank, total128 VGPR/0 AGPR,cap4, no scales.
3. **Reserved:** exactly the compact instruction bytes but256 VGPRs reserved,
   cap2. Read-only ELF comparison confirms whole `.text` byte-identical to
   compact; only resource allocation differs. This isolates the register-cap
   effect without adding scaling arithmetic.
4. **Scaled:** actual192-register triple accumulation plus64 pipeline regs,
   cap2, nonuniform FP32 scales, direct MFMA/VALU register access.

HIP O3/C++20 gfx90a HIP0/BDF31:00.0,CPU+memory NUMA0,block256,automatic clocks,
no model/context/clock override. Useful GEMM FLOPs / full kernel HIP-event time;
extra scale FMAs are overhead,not counted as useful GEMM FLOPs. Packing,H2D,
scale upload,CPU reference/checks excluded. Two nine-launch process repeats,
variant and problem order reversed in repeat2,200 ms selected-kernel warmup.

K32768 TFLOP/s,run1/run2:

|CTAs|Original|Compact128|Reserved256|Scaled256|
|---:|---:|---:|---:|---:|
|104|124.498/124.542|125.302/125.325|125.055/125.212|118.757/118.939|
|208|136.689/136.689|137.375/137.267|137.186/137.240|135.851/135.824|
|312|140.911/140.921|141.416/141.406|139.689/139.689|137.950/138.050|
|1664|147.383/147.401|148.116/148.130|144.905/144.910|142.521/142.512|

M/N respectively1664/1024,3328/1024,4992/1024,4096/6656. Full K8192 and
K32768 results are in `results/scaled/comparison.md`;104/208/312 retained.

- Compact is about0.35-0.64% above original across these long-K grids:
  **no pipeline-depth reduction/performance sacrifice to meet the budget**.
- Reserving256 instead of128 costs little at104/208,but about1.22% at312 and
 2.17% on the large grid. Cap2 also limits concurrent admission; grid counts
  do not establish uniform per-CU placement.
- Adding actual scaling to reserved256 costs about5.02%/1.00%/1.21%/1.65%.
  This includes metadata loads, bank resetting, drain scheduling and final
  scale tail; it is not an isolated FMA latency measurement. In particular,
  the scale load currently joins the group-head LGKM0 rather than being
  prefetched a group ahead. Its separate cost was not isolated or tuned.

## Tests and provenance

**683 checked launches /235 hashed processes**:504 timing launches across
four variants and179 correctness cases. Boundaries cover zero/single stages,
K128 scale boundaries, eight-stage loop threshold, partial groups, long K up
to65536, M/N tails and changed input seeds. Four scale patterns:ones,nonuniform
powers-of-two,non-power-of-two FP32 values,and zeros mixed with fractional
values. N spans multiple scale columns, M spans shared-scale reuse rows.
Nine additional finite-code cases cover all finite B encodings at short K.
Full/sampled refs follow the established size policy; every C gets finite-
after-NaN-poison/guard checking. Every timing launch has raw-clock validation.

The29-stage-count order model independently tracks raw-bank ownership,
per-microtile K8 contribution counts, scale IDs and output history: every K128
group contributes exactly once in ascending order;3N global loads,4N LDS
writes and ceil(N/4) scale loads, including tails/zero. It is NOT a hardware
emulator; actual GPU references and native resource/ISA checks are separate.
Native hot loop has128 MFMA,128 scale FMAs,two scale loads,24 global x4,
32 LDS writes,64 reads,16 barriers,24 waits and one eight-stage backedge.
No accumulator transfers, scratch accesses or AGPR allocation anywhere.

```sh
bash research/native-gemm/build_scaled.sh compact <fresh-build>
bash research/native-gemm/build_scaled.sh heavy <fresh-build>
bash research/native-gemm/build_scaled.sh scaled <fresh-build>
python3 research/native-gemm/run_scaled.py --output <fresh-results>
python3 research/native-gemm/audit_scaled.py --output <fresh-audit>
```

Defaults use `temp/native-gemm/scaled/{compact,heavy,scaled}`. Actual
compilation sources, resource/native manifests, raw clocks/results, commands,
drivers and immutable before-source snapshot: `results/scaled/`;
order/byte-identical control proof: `results/scaled-audit/`.
HIP builds are warning-free; Python/shell syntax and LSP diagnostics pass.
Clangd main-file check reports0 errors. Header check has no compiler/AST
errors but reports three interactive DefineOutline-tweak failures (no suitable
implementation file); the distinction is recorded rather than hidden.

No measured executable overwritten. Source aliases are resolved against each
recorded source snapshot, never today's changed live generator. No objcopy.
Normal unscaled FP16/FP8 build defaults move to fresh scale-host-bridge paths;
original generators/algorithms remain supported and historical drivers pin
original sources. No production/CMake/model/session/system/package/clock change.
