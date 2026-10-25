# Early A loads and balanced q0 MFMA issue

User-proposed scheduling fix, measured as two increments against the late-scale
kernel. No tile/packing/register/fence change and no further tuning sweep.

**Result:** early A loads alone are mixed/slightly negative; the balanced full
schedule improves every tested grid/K. Keep both measured variants, prefer the
balanced schedule. The A loads had been left late to isolate earlier B-prefetch
experiments, not because conversion depended on them.

## Exact producer changes

A stagingv44:51 is consumed by the two A LDS stores. B conversion usesv52:59
and its packed sourcev60:63. Therefore future A loads may issue before the
current B conversion finishes without clobbering any remaining current A use.

1. **earlya:** move ONLY the two future A load instructions ahead of B
   arithmetic. Keep the existing6/18 conversion split and pointer-add positions.
2. **balanced:** issue B and A future loads immediately after eight B
   permutation copies; issue q0/C01,15 conversion VALUs,q0/C10,remaining9
   conversion VALUs,two B LDS stores,adjacent B and A pointer add/carry pairs,
   then q0/C11. q0/C00 remains immediately after headVM0.

```text
LGKM0; retirement barrier; VM0
MFMA q0/C00
A LDS stores x2; B permutations x8; B load x1; A loads x2
MFMA q0/C01
conversion arithmetic x15
MFMA q0/C10
conversion arithmetic x9; B LDS stores x2; pointer adds x4
MFMA q0/C11
LGKM0; publication barrier
unchanged next raw reads,q1 scaling,q2/q3 schedule
```

Actual native full-stage intervening instruction counts:

|Version|C00->C01|C01->C10|C10->C11|
|---|---:|---:|---:|
|Late baseline|19|24|0|
|Early A only|21|22|0|
|Balanced|13|15|15|

All have43 intervening instructions across these gaps. The layout redistributes
work,not removes it. The middle balanced gap is15 VALUs; other gaps mix LDS,
VMEM and SALU. Those are NOT all four-cycle operations, so the counts are a
scheduling heuristic,not proof of60-cycle exact overlap or removed stall time.
No OOO matrix backlog assumption. The last B store-to-publication window is
shorter; its possible wait cost is included in the observed net result.

Pointer add/carry pairs stay adjacent. Bases update after their loads and
before next reuse. Static future-load validity is unchanged; bounded tails
skip loads AND updates when stage+2 does not exist. Startup and final-stage
code are unchanged. The scaled-bank drain still happens at q1 after the new
q0 round,with the original MFMA-result readiness separation.

## Resources / gates

All three versions:256 architected VGPR/0 AGPR/51 SGPR,private0,16 KiB LDS,
reported2 CTAs/CU. Native eight-stage counts identical:128 MFMA,64 packed
scale FMA,24 globalx4,32 LDS writes,64 reads,16 barriers,24 memory waits,
2 scalar scale requests and one backedge. No added copies,NOPs or transfers.

**568 checked launches/232 executable-hashed processes**:190 correctness
launches across two new variants plus378 timings across all three. Existing
logical int64-partial/ordered-std::fma oracle,nonuniform/non-power-of-two/zero
scales,K128/eight-stage boundaries,longK65536,M/N tails,input seeds and
finiteFP8 tests pass. Every C scanned finite after NaN poison,guards checked.
All timed launches include raw-clock metric checks and independent body-span
containment inside HIP-event duration. No reused invalid packed-clock trial.

Checks live under `results/q0-schedule/checks/summary.json`:

-29 stage counts for all variants preserve scale readiness,ordered raw-bank
  contributions and complete group sums.3N weight loads/4N stores/8N reads.
- Instruction-level staging ownership rejects reload before its current LDS
  consumer,copy from pending/wrong B stage,or B reload before all eight copies.
  Conversion destinations are independent of pending future B raw loads.
- A/B pointer streams,ordered conversions/stores and compute/read/fence/scale
  projections match baseline;executed instruction multisets identical.
- Early-A full generated and native flow becomes EXACTLY baseline after only
  restoring those load pairs to their original positions. Balanced native
  semantic projections and resource/opcode budgets match; native gap checks
  assert13/15/15. This is not a latency/memory-return emulator; GPU references
  are a separate correctness gate.

## Matched results

HIP O3/C++20 gfx90a HIP0/BDF31:00.0 CPU+memoryNUMA0,block256,automatic clocks,
no model/context/clock override. Full useful GEMM FLOPs/full HIP event;
packing,H2D,metadata upload,CPU reference excluded. Two counterbalanced nine-
launch process medians after200ms selected-kernel warmup. Reverse variant and
problem order in repeat2. Retain104/208/312 and1664-CTA large regression case.

K32768 TFLOP/s,run1/run2:

|CTAs|Late baseline|Early A only|Balanced|
|---:|---:|---:|---:|
|104|125.618/125.528|125.573/125.663|127.920/127.874|
|208|137.348/137.321|136.890/136.944|139.003/139.072|
|312|139.708/139.773|139.243/139.225|141.397/141.406|
|1664|144.745/144.749|144.026/144.024|146.174/146.176|

Balanced gains~1.85/1.24/1.19/0.99% respectively. Early-A alone is roughly
flat at104 and loses~0.30/0.36/0.50% at208/312/large. Thus this run does NOT
support claiming early-load latency alone explains the full-schedule gain.

K8192:

|CTAs|Late baseline|Early A only|Balanced|
|---:|---:|---:|---:|
|104|117.814/117.418|117.576/117.815|119.345/119.101|
|208|131.885/131.885|131.487/131.537|133.448/133.499|
|312|135.258/135.258|134.979/134.979|137.065/136.957|

M/N respectively1664/1024,3328/1024,4992/1024,4096/6656. Grid count does not
prove uniform per-CU placement;the cap2 kernel admits at most208 concurrent
CTAs across104CUs. No standalone instruction-latency measurement or claimed
statistical significance for tiny near-noise differences.

## Replay / records

```sh
bash research/native-gemm/build_scaled.sh scaled <fresh-earlya> all earlya
bash research/native-gemm/build_scaled.sh scaled <fresh-balanced> all balanced
python3 research/native-gemm/run_q0_schedule.py --earlya-build <fresh-earlya> --balanced-build <fresh-balanced> --output <fresh-results>
python3 research/native-gemm/audit_q0_schedule.py --output temp/native-gemm/checks/<fresh-name>
python3 research/native-gemm/verify_artifacts.py
```

Fresh builds under `temp/native-gemm/q0-schedule/{earlya,balanced}`;actual
source/native/resource/driver/command/reference/timing/clock records under
`results/q0-schedule/`,with one recorded `checks/` set. Intermediate check
runs remain in gitignoredtemp. The historical scale-latency runner pins its
completed snapshots. Old binaries/data remain untouched. No production/CMake/
model/session/system/package/clock changes or additional schedule sweep.
