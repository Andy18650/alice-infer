# Native FP16 and block-scaled FP8 GEMM on gfx90a

**Outcome:** about **146.5 TFLOP/s for FP16** and **146.2 TFLOP/s for
FP16 A / block-scaled FP8 B** on one MI250X GCD. FP8 delivers approximately
half the weight storage at nearly FP16 speed. It still executes FP16 MFMA;
there is no doubled arithmetic ceiling. This is the design summary, not the
experiment chronology. Preferred FP8 endpoint: balanced q0, commit `78c7fe9c`.

## 1. Scope and key results

Native HIP/C++20 and gfx90a ISA, no rocBLAS/CK or library reference. FP32
accumulation and row-major C; alpha=1, beta=0, no split-K or atomics.
**Prepacked inputs are the kernel contract.** Packing, H2D, scale upload and
CPU checking are outside timing. These are standalone kernels, not model-level
prefill results or production engine integration.

Measurements: HIP0/BDF0000:31:00.0, 104 CUs, CPU+memory NUMA0, automatic
clocks, no model/context or clock override. Throughput is useful `2*M*N*K`
divided by the **full HIP kernel-event time**, including setup/output/scaling.

| M x N, K=32768 | CTAs | FP16 TFLOP/s | Final scaled FP8 TFLOP/s |
|---|---:|---:|---:|
| 1664 x 1024 | 104 | 127.4 | 127.9 |
| 3328 x 1024 | 208 | 137.9 | 139.0 |
| 4992 x 1024 | 312 | 141.6 | 141.4 |
| 4096 x 6656 | 1664 | 146.5 | 146.2 |

Means of two process medians, nine checked eager launches each. Columns come
from separate recorded series, **not a fresh FP16-versus-FP8 randomized A/B**:
conclude near-parity, not a proven sub-percent FP8 advantage. K8192 controls
also improved. An unscaled FP8 control reached 148.1, but omitting scales is
not a correct replacement for block-scaled weights.

The practical large-grid band is roughly 144–148 TFLOP/s here. Operand values
and workload change sustained clocks; a faster constant-register MFMA probe
is not a fixed GEMM ceiling. Normalize using measured core/steady clocks,
not nominal 1700 MHz. Never mix absolute core timestamps across CUs.

## 2. Geometry, packing and operand pipeline

**128x128 output tile, K32 stage, four wave64s, one 16 KiB LDS buffer.**
Waves form a 2x2 output grid. Each computes 64x64 through four 32x32x8 MFMA
microtiles: 64 FP32 accumulator values per lane.

128 doubles input/LDS-read reuse versus the earlier 64x64 CTA. The 256x256
candidate had more reuse but excessive register footprint, lower residency
and fewer jobs at small M/N; it lost the comparisons. K64 staging regressed.
These are measured choices, not universal geometry laws. A losing serialized
register-staging control did not invalidate register staging: the winning
producer uses wide global-to-VGPR-to-LDS transfers.

A and B are separate allocations, with format-specific packing:

- **FP16:** two waves produce A's 64-row halves, two produce B's 64-column
  halves; four x4 loads per wave/stage.
- **FP8:** wave `w` produces K8 slice `w` of both operands. Repack A too:
  **two A x4 loads + one B x4 load per wave**, equal conversion work, no
  producer-role branches. Both paths publish the same FP16 LDS layout.

Each operand serves two compute waves and two microtiles within each wave.
Per CTA/K32: 1,048,576 useful FLOPs, 16 KiB LDS writes, 32 KiB LDS reads.
Requested global input falls from 16 KiB to 12 KiB (8 KiB A + 4 KiB B):
**25% less combined input**, not a physical-HBM traffic measurement.

Five eight-register slots hold four current K8 slices plus a spare q3:

1. Complete current LDS reads; join at a **retirement barrier** before reuse.
2. VM0 establishes producer-data readiness. Start a current MFMA immediately.
3. Store next-stage operands in LDS and prefetch future data into freed
   producer registers while executing current matrix work.
4. LGKM0 plus a **publication barrier** establishes next LDS contents.
5. Refill consumed q0/q1/q2 slots and spare q3 while completing current math.
   Alternating q3 slots avoid copies; bounded tails prevent weight overfetch.

A barrier does not replace a completion wait. Nor does a memory wait prove
MFMA completion. **gfx90a MFMA-result-to-VALU hazards need software spacing**;
same-opcode MFMA SrcC forwarding is a separate rule. Do not assume compiler
hazard insertion inside opaque asm, or an out-of-order MFMA backlog.

## 3. FP8 conversion, scaling and register budget

B is **finite E4M3FN, bias7**, not FNUZ; NaNs are outside the contract.
Producer conversion is `permute -> packed signed shift -> mask -> packed
FP16 multiply by256`: four instructions per two values, 32 VALUs for the
16 FP8 values loaded per lane/stage. Explicit IEEE denormal mode preserves
intermediate subnormals; GPU bit tests verify the actual conversion. No output x256 compensation is needed.

All eight permutation copies precede reloading packed v60:63; expanded B in
v52:59 is independent of that future load. A staging v44:51 is reusable after
its LDS stores. These lifetimes—not a generic load-all/compute-all sequence—
permit early prefetch.

For logical scales `s[K128][N128]`:

```text
B_real[k,n] = decode(B_fp8[k,n]) * s[k/128,n/128]
P_g        = FP32 MFMA partial over K128 group g
scaled_C   = FP32_fma(P_g, s[g,n_tile], scaled_C)   // ascending K groups
```

One final full-K scale is wrong when scales vary along K. Four K32 stages
form a group; scales are wave-uniform SGPR values. Two raw banks ping-pong;
new-group q1 MFMAs overlap draining the previous bank into scaled accumulation.
The new q0 round supplies completion separation from the old bank's writes.
A two-K128-group steady body restores static bank assignment without copies;
partial groups flush explicitly. K need not be divisible by256.

FP16 uses one 64-value AGPR bank per lane; its report is 136 VGPR/64 AGPR/46
SGPR with three CTAs/CU. Adding scaled accumulation naively required about 272
allocated registers. It fits **256 architected
VGPR /0 AGPR /51 SGPR, private0, two CTAs/CU**, without shrinking the pipeline:

| VGPR range | Purpose |
|---|---|
| v0:39 | Five operand slots |
| v40:43 | Shared global lane offset; three persistent LDS addresses |
| v44:63 | A payload, expanded B, packed B buffer |
| v64:127 | Persistent scaled accumulation |
| v128:191; v192:255 | Two raw MFMA banks |

Use SGPR global bases and SALU pointer increments. Pass only SGPR inputs across
the opaque asm boundary; store C/timestamps inside it and reuse dead operands
in the epilogue. This removes compiler-owned vector lifetimes, reducing other
state from 80 allocated registers to 64. The compact unscaled control showed
no performance sacrifice. Resource reports can include AGPR allocation in
VGPR totals: **do not sum fields blindly**. Capacity is not a placement proof;
312 CTAs exceed the scaled kernel's 208-CTA concurrent capacity on 104 CUs.

## 4. Scheduling findings

- **Remove redundant control:** peeled FP16 tails reduced its hot two-stage
  loop from 90 to 81 instructions without changing executed data/math order.
- **Hide only dependencies that matter now:** a scale load immediately before
  head LGKM0 forced an unnecessary stall. Loading after the previous-bank q1
  drain, with later explicit readiness, gained 4.11% at 104 CTAs/K32768 and
  0.83% on the large grid. Final single-stage groups need a flush-time wait.
- **Balance issue windows:** final q0 gaps are 13/15/15 native instructions
  instead of 19/24/0. Early A loads alone were slightly negative; the full
  schedule gained 1.85/1.24/1.19/0.99% at 104/208/312/1664 CTAs. Mixed
  SALU/LDS/VMEM counts are a heuristic, not an exact four-cycle latency model.
- **Do not promote a probe into a hardware law:** pure-load controls disproved
  a universal32 B/core-cycle receive ceiling; load width/instruction count
  matters, and requested bytes are not physical HBM traffic.
- **Keep negative results:** pair-batched scales, earlier q1 LDS refill and
  narrow VGPR-pair/LDS-XOR controls had no consistent extra benefit. Larger
  tiles, LDS allocations, queues and lookahead were not automatic wins.
- **Packed operations help, not magically:** packed FP32 scale FMAs halved
  their instruction count and gained about 0.6–1.3%; moves alone were noise.
  Much arithmetic already occupied matrix execution windows.

## 5. Validation and experience to retain

References use original logical matrices, not the GPU packing decoder. Exact
FP16/default FP8 gates use integer/16 inputs and bounded K; scale references
form integer partials then ordered `std::fma` updates. GPU conversion tests
cover all 254 finite codes, signed zero and subnormals. Large GEMMs use sampled
dots plus **all-output NaN-poison/finite and guard checks**—not full numerical
coverage or arbitrary-IEEE certification. The final schedule comparison passed
568 checked launches, including scale variation, seeds and boundary cases.

A consequential bug: **V_PK_MOV defaults duplicated timestamp low words**.
Ratio-only checks and an auditor sharing that wrong assumption both missed
it. Explicit `op_sel:[0,1]`, independent asymmetric64-bit hardware tests with
negative controls, and body-span containment inside HIP-event duration fixed
validation. The first packed clock trial remains INVALID; use the corrected
rerun. Packed FMA also needs explicit scale broadcast selectors: current s14
and previous s15 share aligned s[14:15], selecting the same word for both halves.

For subsequent work:

- Change one hypothesis at a time; retain 104/208/312 and a large-grid control.
- Inspect the **actual embedded ISA**, resources and register ownership. Bounded
  check models supplement hardware tests; they are not formal GPU proofs.
- Use fresh binary paths and executable fingerprints. Never rebuild historical
  binaries merely to make old hashes pass; preserve corrections and failures.
- Keep intermediate checks in gitignored `temp/`, one recorded `checks/` set
  inside its owning experiment. Historical snapshots remain byte-for-byte as-run.
- Prioritize small-grid latency over a fractional large-grid record. Production
  still needs activation packing, model-level gates and end-to-end cost tests.

## Entry points and evidence

FP16: `generate.py`, `data.hpp`. FP8 producer: `generate_fp8.py`. Scaled banks/
register budget: `generate_scaled.py`, `scaled_kernel.hpp`, `data_scaled.hpp`.
Preferred schedule: `generate_scale_latency.py`, `generate_q0_schedule.py`.

Build FP16: `bash research/native-gemm/build.sh 128 <fresh-build>`.
Build scaled FP8: `bash research/native-gemm/build_scaled.sh scaled <fresh-build> all balanced`.

Final results/replay: [Q0_SCHEDULE.md](Q0_SCHEDULE.md). Details:
[SCALED_FP8.md](SCALED_FP8.md), [SCALE_LATENCY.md](SCALE_LATENCY.md),
[PACKED_FP32.md](PACKED_FP32.md). Earlier decisions:
[DESIGN_LOG.md](DESIGN_LOG.md); review/artifact workflow: [AUDIT.md](AUDIT.md).
