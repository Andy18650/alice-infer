# One-CTA 64x64xK FP16 GEMM, direct-to-LDS only

One CTA, **256 threads/four waves in a 2x2 M/N grid**, native
`v_mfma_f32_32x32x8f16`, FP16 operands and FP32 accumulation/output. No FP8,
no dtype conversion in loading/compute, no vendor GEMM, no register-staging
producer. The losing [register baseline](REGISTER_BASELINE.md) is archived
at checkpoint **3217640**, not an active option.

**Current StageK8 path:** [rolling buffers/current-slot retirement](CURRENT_RETIREMENT.md),
2/4/8/16/32 LDS slots and an outer loop explicitly unrolled by the slot count.
Bulk uses lgkmcnt(2) before MFMA and leaves the next pair pending;16 slots
measure ~109 cycles/step. The [older persistent schedule](STAGE8.md) and
[ring-size follow-up](RING_GROWTH.md) are retained history, not the current
retirement protocol. No per-record inner compute loop. The [initial slow protocol](STAGE8_INITIAL.md)
and its measured source/results are retained separately.

## Timing units — primary metric is cycles/MFMA step

Two different timing intervals are recorded:

| Field | What it measures |
|---|---|
| `cta_cycles` | Internal core-clock span: max(wave end)-min(wave begin) |
| `cycles_per_mfma` | That span divided by executed K8 steps per wave |
| `body_us` | The same body region using steady-clock ticks, in microseconds |
| `launch_us` | HIP event interval surrounding the full kernel dispatch |

**Body region:** starts before the first global->LDS stage and stops after
the last stage's synchronization. Includes producer issue/transfer waits,
LDS reads, MFMA pipeline and barriers. Excludes final accumulator-harvest
guard and output stores. The event interval includes that kernel epilogue.
Neither includes CPU input construction/packing, H2D upload, host verification
or the separate one-off staging-copy correctness kernel. These are not CPU
end-to-end or row-major-pointer API timings.

The cycle metric uses the measured core counter directly, not microseconds
multiplied by an assumed frequency. `core_MHz` cross-checks core/steady clocks.

```text
executed_steps = ceil(K/StageK) * StageK/8
cycles_per_mfma = CTA core span / executed_steps
```

One **step** means one MFMA in **each of the four waves**. This preserves the
~64-cycle floor convention of prior rounds; dividing by the aggregate four-
wave instruction count would give a different metric. For K32768, all tested
StageK values execute4096 steps/wave (16384 hardware MFMAs total).
The result is **amortized whole-body cost**, not latency of one MFMA. Tail
padding counts as executed work in this divisor; useful GFLOP/s uses actual K.

## 2x2 wave sharing and layout

| Wave | Output M/N tile | Producer role |
|---:|---|---|
| 0 | (0,0) | A rows0..31 |
| 1 | (0,1) | A rows32..63 |
| 2 | (1,0) | B columns0..31 |
| 3 | (1,1) | B columns32..63 |

All waves compute. A0 serves waves0/1, A1 serves2/3; B0 serves0/2, B1 serves1/3.
A K8 record contains `[A0:128,A1:128,B0:128,B1:128]` dwords, **2048 staged
bytes**, while nominal LDS-to-register requests remain4096 B/four-wave step.

```text
A byte address = record*2048 + M_tile*512 + lane*8
B byte address = record*2048 + 1024 + N_tile*512 + lane*8
```

Consumers use two ds_read_b64 instructions, wait on lgkmcnt, and feed MFMA
directly from raw destinations. No ready banks/copies. Depth1..8 keeps
lgkmcnt(2*(D-1)) <=14. Bank=dword_address%32: block bases are bank-aligned,
and consecutive b64 pairs distribute aggregate pressure over32 banks.
This is not a bank-conflict counter or exact read-phase proof.

## Input contract and chunks

The host constructs row-major **A[64,K], B[K,64] FP16 inputs**, then bit-
rearranges them into the layout above. The GPU consumes packed input;
**packing/upload are outside timing**, and no end-to-end speed is claimed.
No FP8 or value/dtype conversion is involved in that rearrangement.
K may be zero, odd or tailed; packing zero-pads to StageK and the reference
contracts only actual K. C is FP32 row-major64x64, alpha1/beta0.

StageK is the number of K elements staged per chunk. A+B need256*StageK B:

| StageK | One LDS buffer | Two LDS buffers |
|---:|---:|---:|
| 32 | 8 KiB | 16 KiB |
| 64 | 16 KiB | 32 KiB |
| 128 | 32 KiB | 64 KiB |

StageK8 additionally uses2/4/8/16/32 slots (4/8/16/32/64 KiB), raw depth2 and fixed
register parity/LDS offsets in its unrolled outer pipeline. The table above
covers larger chunks' one/two-buffer implementation. These are measured
choices, not architectural limits. StageK changes how many dword copies are batched,
not their width. A producer wave moves StageK bytes/lane: StageK/4 dword
instructions, with two waves staging A and two staging B.

## Direct copy and synchronization

Actual gfx90a **`global_load_dword ... lds`** bypasses payload VGPRs. v[0:1]
is a source address, not a dwordx2 data result. m0 supplies a uniform LDS base;
destination adds lane*4. The immediate offsets0/256/2048/2304 affect both
ends and cover two K8 blocks for larger chunks, then pointer/m0 advance4096 B.
StageK8 uses just0/256 and advances2048 B in its persistent pipeline. The asm saves/
restores reserved m0 and includes its write wait state. There is no DS write
or global->VGPR->LDS payload path. `--producer` has been removed.

Global direct completion uses **vmcnt**, LDS reads **lgkmcnt**. The CTA
barrier publishes all writers' completed data across waves and retires reads
before buffer reuse. No CUDA cp.async/mbarrier model is assumed.

```text
produce tile0 -> buf0; vmcnt(0); CTA barrier
issue tile1 -> buf1 asynchronously
compute tile0; drain LDS reads
vmcnt(0); CTA barrier          // tile1 ready, tile0 storage reusable
issue tile2 -> buf0; compute tile1; ...
```

The single-buffer path waits/publishes, computes, drains/barriers, then refills
that same buffer. Double buffering overlaps return latency; whole next-tile
issue still precedes current compute, so it does not hide all issue work.
In the larger-chunk path, raw LDS priming/drain occur once per compute stage,
with unused lookahead
wrapping safely inside the stage. Accumulators stay in AGPRs across stages.
Ending timestamps precede the conservative accumulator-harvest guard.

ISA and copy checks confirm no hidden VMEM drain/barrier between future
issue and current MFMA; completion is waited before the next stage transition.
Sources: [gfx90a assembler syntax](https://releases.llvm.org/18.1.6/docs/AMDGPU/AMDGPUAsmGFX90a.html),
[LLVM gfx90a global-load-LDS lowering](https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.global.load.lds.ll).
The local assembler accepts the dword LDS form, not wider x2/x4 forms.

## Current validation and cycles

**Current larger-chunk regression:340 launches pass exact all4096-C/32-guard checks**:196 verification cases
and144 warmed timing launches. Every GEMM case also checks each dword in both
LDS buffers. The reference uses original row-major integer units, not packed
indices/checksums. Inputs are integers/16; supported verification bounds
K<=65536/LDS steps<=8192 keep all sums exact in FP32. Negative controls reject
value/guard corruption. Arbitrary FP16 NaN/Inf/subnormal behavior is not tested.

Coverage includes K0/odd/boundaries/tails through65536, three StageK choices,
depths1/2/3/4, one/two buffers and independent A/B seed changes. Pure-LDS
2x2 sharing tests passed before GEMM was added and remain modes for diagnosis.
Six builds/30 entries have no scratch/static-LDS/lane spills, no inner
accumulator copies or dtype conversions. All **24 surviving benchmark core
instruction/register lists match** checkpoint3217640 after address/branch
normalization. Old `--producer reg|direct` values are both rejected.
The first post-cull runner mistakenly used archived executable paths; those
results are explicitly flagged in `results/direct-only/PROVENANCE.md`.
Corrected current-binary runs verify source/binary hashes before launching.
The original StageK8 persistent sweep separately passed405 checked launches.
The current rolling-retirement study passes882 checked samples, including720
new verification cases and matched timing controls; see its linked note for
lifetime/ISA checks, all-stage validation and explicit timeout recovery.

Native HIP O3/C++20, gfx90a, HIP0 / PCI31:00.0, **CPU+memory NUMA0**, one CTA,
K32768, D2/U4, no model/context/clock changes. Individually warmed200 ms, two
randomized process-order repeats, nine checked samples each.
**Primary: whole-body core cycles per four-wave MFMA step, run1/run2:**

| StageK | One LDS buffer | Two LDS buffers |
|---:|---:|---:|
| 32 | 314.887 / 311.806 | 245.812 / 245.469 |
| 64 | 216.014 / 215.243 | 147.018 / 146.644 |
| 128 | 193.604 / 187.580 | **143.557 / 143.020** |

For the last row, body duration is348.000/346.720 us and useful throughput
771.366/774.214 GFLOP/s for **one CTA**, not full-GPU performance. It is not
compute-bound at the ~64-cycle LDS-only floor. No physical traffic/port limit,
variance-controlled optimum, optimal interleaving or end-to-end claim.

## Build / run

```sh
# Larger chunks: D1..8, StageK32/64/128, U divides StageK/8.
# StageK8: build.sh 2 1 8; --buffers 2|4|8|16|32; see RING_GROWTH.md.
bash research/fp16-gemm64/build.sh 2 4 128
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/d2-u4-t128-direct-only
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/d2-u4-t128-direct-only/probe \
  --mode gemm --buffers 2 --k 32768 --rounds 9

# No global operand staging in LDS-only diagnostics:
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/d2-u4-t128-direct-only/probe \
  --mode lds-sweep --records 32 --steps 4096
python3 research/fp16-gemm64/run.py
```

The runner expects D2 StageK32/64/128 plus D1/D3/D4 StageK64, all U4. Current
outputs append `-direct-only`; old binaries/results remain intact. There are
four benchmark kernels/build for larger chunks (two GEMM buffering choices,
two LDS diagnostics) and one test-only copy kernel. StageK8 instead has five
GEMM buffering/unroll choices plus those diagnostics and copy validator.
Runners verify the current sole producer.
Use `--resume` only on an unchanged experiment. A generated compile database
supports clangd. No production/CMake/model-fixture changes.

Current CSV/logs, raw clocks, commands, snapshots, hashes, inventories,
rejections and comparisons: `results/direct-only-persistent-regression/` for
the larger-chunk rerun, `results/retire-current/` for current StageK8.
Earlier StageK8 studies remain at `results/stage8-persistent/` and ring16/ring32.
The original misbound `results/direct-only/` is flagged, not deleted. Pre-cull results and
source are preserved in `results/final/` and checkpoint3217640. Initial grid
checks are in `results/grid/`; development producer snapshots are under
`temp/fp16-gemm64/`. No old controls are silently rerun as active options.
