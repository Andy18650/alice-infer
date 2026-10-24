# FP8 clustered copies, early B prefetch and current-MFMA interleave

Two user-directed increments, measured separately against their preceding
version. No tile/packing/fence/operand-read or datatype changes.128x128x32,
four waves, each wave produces one K8 A/B slice, two A x4 loads + one B x4.
The earlier conversion-burst result remains frozen under `results/fp8-b/`.

## 1. Cluster permutations and immediately reload the freed raw buffer

Original in-place expansion cannot reload its packed buffer early because
converted current-stage values overlap that buffer. Keep the eight expanded
B dwords atv52:59, add a dedicated packed B buffer atv60:63. All eight
permutations readv60:63 and copy to distinctv52:59 before the next B load.
No remaining arithmetic reads the packed source. The next load can finish at
any point after those copies without clobbering current-stage conversion.

This intentionally adds four fixed VGPRs. Metadata140->144 VGPR,64 AGPR/46
SGPR unchanged,private0,16 KiB LDS; runtime reported capacity still3 CTAs/CU.
This first A/B therefore measures a copy/buffering/schedule package, NOT an
isolated measurement of early-load latency or permutation clustering.

Startup still loads stages0 and1 at its old points; no current operands exist
yet for startup interleaving. In ordinary non-final stages, the next B load
is literally the instruction after the eighth permutation. Its pointer
advance follows. Only statically valid stage+2 fetches issue; bounded tails
never load past K. A future loads stay after the stores, isolating the
requested B retiming rather than adding another A prefetch change.

## 2. Distribute conversion arithmetic among current MFMAs

With identical buffers/resources, move existing current-q0/C01 after the first
six conversion arithmetic instructions (two packed output pairs). The other
18 arithmetic instructions follow it. Current C00 already issues immediately
after head VM0. Remaining C10/C11 stay after stores/A prefetch.

```text
LGKM0; retirement barrier; VM0
current q0/C00 MFMA
two A LDS stores
eight permutation copies
next B global x4 load; advance B pointer       [only when stage+2 exists]
six shift/AND/multiply instructions
current q0/C01 MFMA                           [moved from below]
remaining18 shift/AND/multiply instructions
two B LDS stores
next A two global x4 loads; advance A pointer  [only when stage+2 exists]
current q0/C10 MFMA
current q0/C11 MFMA
LGKM0; publication barrier
next raw q0/q3 reads; unchanged q1/q2/q3 math/read pipeline
```

Both new hot loops have exactly147 instructions,32 MFMAs,6 global x4,
8 LDS b128 writes,16 LDS b128 reads,16 EACH perm/shift/AND/multiply,
six waits,four barriers and three control instructions. Same144/64/46
register metadata and no spills. Actual native early->interleaved loop is
exactly the two C01 moves in the two-stage pair; every other instruction
retains its order. Per-accumulator arithmetic order is identical.

Yes: moving C01 earlier leaves only C10/C11 after B stores and shortens the
store-to-publication-wait window. Net measured gains survive that tradeoff.
Individual LGKM wait stalls are NOT instrumented, so no zero-stall or fixed
64/128/192-cycle overlap claim. Wave issue remains in-order; no assumed OOO
matrix backlog. No extra MFMA, altered FLOP count or changed wait/barrier.

## Matched results

Native HIP O3/C++20 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0,block256,
automatic clocks,no model/context/clock override. Full useful FLOPs / full
kernel HIP-event time; packing/H2D/reference excluded. Each A/B uses two
counterbalanced nine-checked-launch process repeats after200 ms kernel warmup.
104/208/312 CTAs retained,plus1664-CTA large regression check.

K32768 useful TFLOP/s, run1/run2. These are **two adjacent matched A/Bs**, not
one three-way randomized trial; the repeated early control is shown twice.

|CTAs|Original burst (step1 before)|Early copies/load (step1 after)|Early control (step2 before)|Interleaved (step2 after)|Final efficiency|
|---:|---:|---:|---:|---:|---:|
|104|120.687/120.896|119.776/119.940|119.796/119.755|124.586/124.631|80.49/80.44%|
|208|134.063/134.179|134.515/134.658|134.619/134.554|136.676/136.796|89.96/90.00%|
|312|139.067/139.141|139.132/139.160|139.141/139.123|140.978/140.864|93.47/93.40%|
|1664|146.494/146.482|146.817/146.821|146.788/146.785|147.333/147.364|98.02/98.01%|

At K8192:

|CTAs|Burst (step1)|Early (step1)|Early control (step2)|Interleaved (step2)|
|---:|---:|---:|---:|---:|
|104|113.228/113.522|112.642/112.643|112.934/112.788|116.633/116.867|
|208|129.103/129.247|129.631/129.969|129.775/130.017|131.339/131.537|
|312|135.398/135.468|135.539/135.679|135.679/135.714|137.136/136.850|

M/N respectively1664/1024,3328/1024,4992/1024 and4096/6656. The first
increment is mixed (small104 regression); the isolated second MFMA move is
positive at every tested grid/K. The combined observations improve on the
original FP8 burst schedule, but are not a fresh matched comparison to FP16.
No uniform per-CU placement is inferred from grid count or capacity.

## Gates / preservation

- Each A/B has312 checked launches:60 functional/seed/boundary cases +252
  timings,88 executable-hashed processes. Both A/Bs:624 checked launches.
- Each new version additionally passes18 full-reference all-finite-FP8 GEMM
  launches atK1/7/8/9/31/32/33/63/64,M129/N131,random signed A=1/16.
- Each GPU bit gate checks4064 vectors/65024 half outputs,all254 finite
  E4M3FN codes every byte position,mixed neighbors,signed zero/subnormals.
  For these schedules it deliberately poisonsv60:63 immediately after all
  copies and BEFORE conversion arithmetic, proving source-buffer independence.
- Independent524288 padded-coordinate packing checks per gate unchanged.
- At29 stage counts: same compute/raw-read/fence projection for all3 versions;
 3N global loads,4N stores,8N copies, no overfetch; exactly stages-1 C01 moves
  for early->interleaved. Native loops/resources match the exact two moves.
  Current explicit `burst` generation reproduces archived burst asm exactly.
  This is a control/order model, not a hardware emulator.
- Default-domain refs remain exact; large GEMMs use sampled original logical
  dots plus all-C finite-after-NaN-poison/guards and raw-clock validation.
  Full finite-code tests retain the exact K<=64 integer oracle,not a new
  arbitrary-range/long-K IEEE numerical claim.

No archived source/data or measured executable overwritten; no objcopy used. Original
FP8 audit/driver pin their completed snapshots now that the live generator
has intentional later changes. Normal FP16 generator and kernel untouched;
FP16 remains the default format. New FP8 defaults use a fresh schedule namespace.

```sh
# Rebuild historical sources into FRESH directories, never overwrite measured builds:
bash research/native-gemm/results/fp8-early/after/source/build_fp8.sh <fresh-early> early
bash research/native-gemm/results/fp8-interleaved/after/source/build_fp8.sh <fresh-interleaved> interleaved
python3 research/native-gemm/run_fp8_schedule.py --step early --after-build <fresh-early> --output <fresh-step1>
python3 research/native-gemm/run_fp8_schedule.py --step interleaved --before-build <fresh-early> --after-build <fresh-interleaved> --output <fresh-step2>
python3 research/native-gemm/audit_fp8_schedule.py --output <fresh-audit>
python3 research/native-gemm/verify_artifacts.py
```

Raw/reference/clock/source/native/driver records: `results/fp8-early/`,
`results/fp8-interleaved/`; order/no-overfetch proof in
`results/fp8-schedule-audit/` and expanded `results/fp8-schedule-audit-final/`.
The first snapshot pass omitted before-side FP8 extra files (the shared driver
previously only needed FP16 before files). Missing files were supplemented from
the immutable parent compilation snapshot, never today's changed generator;
`source-supplement.json` records paths/hashes and all compiled manifest hashes
match. The actual measured driver copy is retained; future driver snapshots
now accept explicit before-side extra sources.
No further interleave-count/tile/wait/packing tuning or production/CMake/model/
system/package/clock-policy changes. Research status appended to research-log.
