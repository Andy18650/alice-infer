# One-CTA64x64xK FP16 GEMM — fixed active winner

Native gfx90a HIP/O3/C++20 research, FP16 operands and FP32 accumulation/output.
One CTA/256 threads, four waves in a2x2 M/N grid. GPU-only math, no vendor GEMM,
FP8 conversion, register-staged producer, ready-bank copies or CPU fallback.
No production/CMake/model-format changes.

## Active matrix

**StageK8, two raw buffers,16 LDS slots, outer-unroll16,32 KiB LDS.**
Paired global-copy offsets0/256 and2048/2304 halve source-pointer advances.
Current-slot retirement, LGKM2 immediately before MFMA, same-buffer read of
n+2 immediately after MFMA, earlier publication at VM24, M0 setup before
the existing barrier and no bulk NOP. Accumulators stay in AGPRs across K.

Only one GEMM is generated/dispatched. See [MATRIX.md](MATRIX.md) for the
result scan, removed cases, smaller default suite and cull verification.
Two LDS-only kernels remain opt-in diagnostics; one kernel validates staging
copies. Historical source, raw data and measured binaries remain intact.
The constants are measured choices, not permanent architectural constraints.

## Build and use

```sh
bash research/fp16-gemm64/build.sh                         # optional output directory, no dimension arguments
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/active
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/active/probe \
  --mode gemm --k 32768 --rounds 9
python3 research/fp16-gemm64/run.py                        # 24 verification cases +18 timed samples
```

`run.py --output <fresh-dir>` preserves prior measurements. Verification is
ON by default. `--full` runs144 winner-only verification cases instead of24;
`--diagnostics` adds four LDS-reference checks. Optional explicit archived
winner/source arguments allow a matched A/B, but no baseline is replayed by
default. Obsolete dimension flags, `--buffers` and the old matrix runner are
removed. Old commands in historical notes apply to those checkpoints only.

For a focused LDS-only check (no global operand staging in its timed body):

```sh
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/active/probe \
  --mode lds-sweep --records 32 --steps 4096 --rounds 1
```

## Layout and input contract

The host constructs ordinary row-major **A[64,K], B[K,64] FP16** inputs, then
bit-rearranges them outside timing into K8 records:

```text
[A0:512 B][A1:512 B][B0:512 B][B1:512 B] =2048 unique staged bytes
```

Waves0/1 stage A0/A1, waves2/3 stage B0/B1; all four compute. A0 serves0/1,
A1 serves2/3, B0 serves0/2, B1 serves1/3. Nominal LDS-register payload is4096 B
per four-wave MFMA step, not a physical HBM/port count. Each wave loads two
DWORDs/lane via native `global_load_dword ... lds`; payload bypasses VGPRs.
M0 is a wave-uniform destination, plus lane*4 and the immediate displacement.
Immediate offsets affect BOTH source and destination, hence paired M0 bases.

K may be zero, odd or tailed; pack zero-pads to K8. C is row-major64x64 FP32,
alpha1/beta0. Packing/upload are excluded: this is not end-to-end row-major-
pointer GEMM performance. The independent integer/dyadic reference contracts
original row-major A/B, checks all4096 outputs and32 guards, and bounds
verification to K<=65536 for exact FP32 sums. No arbitrary IEEE edge-case claim.

## Timing definitions

- `cta_cycles`: max(wave core end)-min(wave core begin), not median-wave time.
- `cycles_per_mfma`: that CTA span / executed K8 steps per wave.
- `body_us`: corresponding region measured by the steady GPU counter.
- `launch_us`: full HIP event interval around the kernel, including epilogue.

At K32768 the divisor is4096 steps/wave, not aggregate16384 MFMAs. Body includes
startup, global/LDS staging, waits, barriers, compute and tail, but excludes
final accumulator-harvest guard/output stores. CPU packing/H2D, copy checks
and host verification are outside both timing intervals. This is amortized
whole-body cost, not isolated MFMA latency. No assumed clock conversion is used.

Native O3/HIP0/gfx90a, CPU+memory NUMA0, one CTA, K32768, no model/context/clock
changes: the pre-cull winner measured93.510/92.646 cycles/step; the cull recheck
93.280/93.116 (227.200/226.720 us). Its1652 timed instruction/register lines
are unchanged after branch/address normalization; no pruning speedup is claimed.
This is one-CTA performance, not full-GPU throughput or a hardware ceiling.

## Preserved research history

- [Initial sharing/GEMM/register control](REGISTER_BASELINE.md), checkpoint3217640.
- [Initial slow K8 scheme](STAGE8_INITIAL.md), then [persistent raw2](STAGE8.md).
- [Ring growth16/32](RING_GROWTH.md), then [partial LGKM/current retirement](CURRENT_RETIREMENT.md).
- [Post-MFMA reads](POST_MFMA_READ.md), [pre-barrier M0](M0_BARRIER.md), [paired global addressing](PAIRED_GLOBAL.md).
- Original post-cull runner attribution correction: `results/direct-only/PROVENANCE.md`.

Historical results stay under `results/`; obsolete source is recoverable from
checkpointb081cef and frozen source snapshots. Current reduced verification,
ISA comparison, rejection checks and selection evidence: `results/active-cull/`.
Fresh build flags, generated header, CDB, disassembly and hashes are under
`temp/fp16-gemm64/active/`. The scoped clangd configuration names that CDB.
