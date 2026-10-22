# One-CTA64x64xK FP16 GEMM — StageK16 experiment

Native gfx90a HIP/O3/C++20 research, FP16 operands and FP32 accumulation/output.
One CTA/256 threads, four waves in a2x2 M/N grid. No FP8, vendor GEMM,
register-staging producer, ready-bank copies, CPU fallback or production change.

## Current experiment / preserved baseline

[StageK16](STAGE16.md) groups two original K8 records per rendezvous and M0
setup. Primary:**8 slots/32 KiB**, preserving the previous K8 winner's LDS
budget and resident K window; also tested:16 slots/64 KiB. Two raw buffers,
64x64 M/N tile, packed layout and b64 reads stay unchanged. B128 is deferred.

Equal-budget timing is mixed (~93 cycles/K8-equivalent step), so no clear
win is claimed.64 KiB gives a small single-CTA gain (~92 cycles), not a
prefill/CTA-residency recommendation. See the experiment note for both
matched repeats, verification and assembly. No unrelated tuning is included.

The prior [result-led cull](MATRIX.md) selected StageK8/raw2/16 slots, paired
global addressing and pre-barrier M0. That frozen binary/source remain at
`temp/fp16-gemm64/active/` and `results/active-cull/source/`; they are not
overwritten or silently rebuilt by these new candidates.

One GEMM is generated per build; the only choice is the explicitly authorized
8/16-slot StageK16 experiment. Two LDS-only kernels are opt-in diagnostics;
one kernel checks every staging word/guard. There is no broad matrix runner.

## Build / run

```sh
bash research/fp16-gemm64/build.sh                  # defaults to K16,8 slots/32 KiB
# Optional output path and8|16 slots:
bash research/fp16-gemm64/build.sh temp/fp16-gemm64/my-k16-s16 16
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/stage16-s8
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/stage16-s8/probe \
  --k 32768 --rounds 9
python3 research/fp16-gemm64/run.py --output <fresh-results-directory>
```

Default runner retains24 focused verification cases plus18 timed samples.
`--full` selects144 candidate-only verification cases; `--diagnostics` adds
four independent LDS checks. Explicit frozen baseline/source arguments allow
matched A/B, but no baseline runs by default. The previous fixed K8 winner
needs no legacy dimension flag. Runtime `--buffers`, obsolete broad build/
generator axes and old matrix runner options remain absent. Current build
signature is `[output-directory] [8|16 slots]`. Old notes' commands apply to
their checkpoints, not current source. Output paths must be fresh.

## Layout / schedule

CPU constructs ordinary row-major FP16 A[64,K] and B[K,64], then bit-rearranges
outside timing into unchanged2 KiB K8 records `[A0,A1,B0,B1]`. Each K16 stage
contains two records,4 KiB. Waves0/1 produce A0/A1, waves2/3 B0/B1; all compute.
A/B panels are each shared by two waves. Current stage16 producer uses four
DWORD instructions/lane at offsets0/256/2048/2304, one M0 setup/restore and
one4096-byte pointer advance. Payload bypasses VGPRs.

Each wave's two MFMAs use raw0/raw1. After each, two ds_read_b64 instructions
prefetch that fragment of the next K16 stage. LGKM2 precedes MFMA; only the
last stage's second MFMA drains LGKM. VM20/52 publishes the stage after next;
M0 setup is before the existing CTA barrier, then current-slot refill occurs.
No barrier between the two MFMAs, no inner compute loop or extra bulk NOP.

K0/odd/tails are supported via zero-padding to16; C is FP32 row-major64x64,
alpha1/beta0. Reference uses original row-major integer units, checking all
4096 outputs and32 guards. Verification bounds K<=65536 for exact FP32 sums;
arbitrary FP16 IEEE edge cases are not certified. Packing/H2D and verification
are outside timing: no end-to-end row-major-pointer API claim.

## Timing

`cta_cycles` is max(wave core end)-min(wave core begin). `cycles_per_mfma`
divides by executed MFMAs/wave, **two per K16 stage**. At K32768 the divisor
is4096, not2048 stages or aggregate16384 instructions. `body_us` uses the
steady GPU clock over startup/compute/staging/waits/barriers/tail; final
accumulator-harvest/output stores are outside. `launch_us` is the full HIP
event interval. No assumed clock conversion or full-GPU throughput claim.

Native O3 HIP0/gfx90a, CPU+memory NUMA0, one CTA/256 threads, K32768, no model/
context/clock changes:32 KiB candidate92.821/94.006 versus K8 baseline93.452/
92.976;64 KiB candidate92.241/91.959 versus93.271/93.276. These are two warmed
process medians of nine checked samples each. Both experiments pass120 checked
samples total plus18 targeted short-fill/tail smokes. No further optimization.

## History

[MATRIX.md](MATRIX.md) documents the pre-experiment cull;
[PAIRED_GLOBAL.md](PAIRED_GLOBAL.md), [M0_BARRIER.md](M0_BARRIER.md),
[POST_MFMA_READ.md](POST_MFMA_READ.md), [CURRENT_RETIREMENT.md](CURRENT_RETIREMENT.md),
[RING_GROWTH.md](RING_GROWTH.md), [STAGE8.md](STAGE8.md),
[STAGE8_INITIAL.md](STAGE8_INITIAL.md) and [REGISTER_BASELINE.md](REGISTER_BASELINE.md)
preserve the earlier evidence. Unsafe [ablation results](ABLATIONS.md) are
historical performance-only K8 diagnostics, not current valid candidates.

Current data/source/native extracts/hashes are `results/stage16-s8/` and
`results/stage16-s16/`; older records/binaries remain intact. Scoped clangd
uses `temp/fp16-gemm64/stage16-s8/`. No b128/data.hpp, CMake/model-format,
package/system/clock-policy or production changes.
