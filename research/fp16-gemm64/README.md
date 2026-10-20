# 2x2 shared-LDS waves -> one-CTA 64x64xK FP16 GEMM

Standalone gfx90a research. One CTA, **256 threads/four waves**, native
`v_mfma_f32_32x32x8f16`, FP16 inputs and FP32 accumulation/output. No FP8,
no dtype conversion in loading/compute, no vendor GEMM. The previous
[operand-delivery probe](../fp16-load-matmul/README.md) is untouched.

## 1. Shared-LDS 2x2 mapping, verified before adding GEMM

| Wave | Output tile (M,N) | Cooperative producer role |
|---:|---|---|
| 0 | (0,0) | A rows0..31 |
| 1 | (0,1) | A rows32..63 |
| 2 | (1,0) | B columns0..31 |
| 3 | (1,1) | B columns32..63 |

All waves compute. The producer role need not match the A/B slice that wave
consumes. A0 is shared by waves0/1, A1 by2/3; B0 by0/2, B1 by1/3.
Each native K8 step has **2048 unique staged bytes**, versus4096 in four
private A+B streams. The four waves still request **4096 bytes into VGPRs**
from LDS; do not claim that cross-wave sharing halves register delivery.

LDS record layout, in dwords:

```text
[K8 record][A0:128][A1:128][B0:128][B1:128] = 512 dwords = 2048 bytes
A address = record*2048 + M_tile*512 + lane*8
B address = record*2048 + 1024 + N_tile*512 + lane*8
```

Each b64 gives four FP16 elements for one MFMA operand. Consumers use two
independent `ds_read_b64` addresses and direct raw-ring consumption with
mandatory lgkmcnt(2*(D-1)). There are no ready buffers/copy VALUs. D1..8
keeps that immediate <=14. D/U rotation uses D/gcd(D,U) phase bodies.

The bank index is dword_address%32. Block/record bases are bank-aligned;
within a b64 the pair is at banks(2*lane)%32 and the next bank, aggregate
balanced over32 banks. No artificial padding/stride is added. This describes
bank balance, not a hardware conflict-counter or exact read-phase proof.

Phase1 initializes this shared layout locally and verifies the **full row-major
64x64 C**, rather than four unrelated tile products. The initial26 launches
passed before GEMM was added (hot/full/odd-window sweep, independent A/B seed
changes); original binary/source are retained in `temp/fp16-gemm64/phase1/`.
The final suite repeats and extends those checks. Its hot LDS rate is ~64
cycles/MFMA; swept addresses add some instruction/entry cost.

## 2. Actual GEMM and input contract

The harness constructs ordinary row-major **A[64,K] and B[K,64] FP16 inputs**,
then bit-rearranges them into the common packed layout above. Both GPU staging
paths consume the **same packed input**. This is not a row-major device-pointer
GEMM API: **packing, input construction and H2D upload are outside timing**.
No packed-format speed is presented as end-to-end row-major GEMM speed.
Packing changes layout only, not dtype; no FP8 is involved.

K is arbitrary (including0/odd/tails). Input packing zero-pads to StageK;
the kernel computes whole stages. The independent CPU reference contracts
only the actual K. C is FP32 row-major64x64, alpha1/beta0, guarded and poisoned.

StageK is build-time32/64/128. One stage uses256*StageK bytes:

| StageK | One LDS stage | Two stages |
|---:|---:|---:|
| 32 | 8 KiB | 16 KiB |
| 64 | 16 KiB | 32 KiB |
| 128 | 32 KiB | 64 KiB |

Each of the first two waves stages one A block; the other two stage one B
block. Every wave owns equal staging volume and later computes its tile.

### Register-staged control (`--producer reg`)

All coalesced dwordx4 loads for a wave's stage are batched in one asm block,
then vmcnt(0), then ds_write_b128. Every pending payload register remains
private/live inside that block. No premature compiler copies can precede the
wait. Producer LDS writes are drained before partial LDS-read waits, avoiding
mixed read/write LGKM accounting.

This is a **non-overlapping producer control**. Its two-buffer mode does not
pretend to be an optimized asynchronous VGPR-prefetch pipeline; globals are
completed and LDS written before current computation. It is not a universal
best register-staging result.

### Native direct-to-LDS (`--producer direct`)

Actual gfx90a instruction: **`global_load_dword ... lds`**, not the newer
`global_load_lds_dword` mnemonic. m0 supplies a wave-uniform LDS byte base;
the destination adds lane*4. The signed immediate offset affects **both**
source and destination. Four coalesced reads at offsets0/256/2048/2304 copy
two K8 blocks; pointer and m0 then advance4096 bytes. No payload VGPR or
DS-write instruction is used by this producer.

The producer saves/restores compiler-reserved m0 once per stage in the same
asm block, and includes the gfx90a m0-write wait state. It does not list a
reserved register as a destructively clobbered ABI value. Global completion
is tracked with **vmcnt**, not the consumers' lgkmcnt. All source addresses
are coalesced; direct-to-LDS does not require uncoalesced row-major gathers
because the supplied input is packed.

Assembler evidence and sources:
- [LLVM gfx90a instruction syntax](https://releases.llvm.org/18.1.6/docs/AMDGPU/AMDGPUAsmGFX90a.html)
- [LLVM gfx90a global-load-LDS lowering tests](https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.global.load.lds.ll)
- Local `llvm-mc-21 -mcpu=gfx90a` accepts the native form. Numerical tests
  also check every transferred dword in **both** LDS buffers, including m0
  offsets at the 64 KiB boundary. Do not assume newer builtin availability
  tables imply the same spelling or frontend support on CDNA2.

## 3. Synchronization and overlap

No CUDA cp.async API/mbarrier model is assumed. Both producers have explicit
completion and **whole-CTA barriers** for cross-wave publication and reuse.

Single-buffer:

```text
produce tile0; finish producer; CTA barrier
compute tile0; drain all LDS reads; CTA barrier
produce tile1 into reused buffer; finish; CTA barrier
...
```

Double-buffer direct:

```text
produce tile0 -> buffer0; vmcnt(0); CTA barrier
issue tile1 -> buffer1 asynchronously
compute tile0 (LGKM reads only)
vmcnt(0); CTA barrier     // next published AND current reads retired
issue tile2 -> buffer0
compute tile1
...
```

All four waves execute barriers uniformly. Compute explicitly drains its LDS
read queue. The next-stage VMEM wait is per producing wave, and the CTA barrier
collects all writers before readers use the next buffer. No consume/produce
flags or token counts are invented.

Actual-code-object inspection confirms future direct VMEM issue occurs before
current MFMA with **no VMEM drain or barrier in between**, and an explicit
VMEM completion wait follows computation before the stage transition. The
pipeline overlaps return latency, not all issue overhead: whole next-tile
issue still precedes the current compute block. This is a correct first GEMM,
not a claim of compute-bound or optimal interleaving.

The direct LDS->MFMA pipeline primes/drains once per compute stage; unused
lookahead wraps within the stage so it cannot read an unproduced next stage.
Accumulators stay in AGPRs across stages. Whole-core timing starts before
initial producer work and ends after stage synchronization; a final harvest
guard/output stores follow the ending timestamps. Packing is excluded, but
all measured producer/consumer/barrier work is included.

## 4. Verification and measured results

**620 launches pass exact all4096-output/32-guard checks**. There are368
verification-only cases and252 warmed timing launches. Every GEMM invocation
also verifies every dword of a separate two-buffer copy using the selected
producer. Host reference uses original row-major integer units, not packed
indices or GPU output checksums. Constructed inputs are integers/16; K<=65536
and LDS steps<=8192 keep all sums exactly representable in FP32. Checker
negative controls reject value/guard corruption. This does not cover arbitrary
FP16 NaNs/infinities/subnormal semantics.

Coverage: K0,1,2,7,8,9,31,32,33,63,64,65,127,128,129,255,257,1024,32768,65536;
three StageK values; depths1/2/3/4; one/two buffers; both producers; independent
A-only/B-only/both changes. LDS-only checks include odd seven-record windows
and short partial-phase exits. Six builds/48 kernel entries have zero scratch/
static-LDS/lane spills; no accumulator reads/writes inside timed computation,
no dtype conversion instructions, and direct wait->MFMA consumption.

Native HIP O3/C++20, gfx90a, HIP0 / PCI31:00.0, **CPU+memory NUMA0**, one CTA,
K32768, D2/U4, no LLM model/context or clock/power overrides. Per-configuration
warm-up200 ms, two randomized process-order runs, nine checked samples each.
**Whole packed-input GEMM**, median core cycles per executed MFMA (run1/run2):

| StageK | Reg, one buffer | Reg, two buffers | Direct, one buffer | Direct, two buffers |
|---:|---:|---:|---:|---:|
| 32 | 333.441/334.229 | 342.267/340.271 | 312.198/312.561 | 247.816/247.437 |
| 64 | 245.847/240.451 | 246.861/244.389 | 214.424/214.378 | 147.353/146.988 |
| 128 | 213.188/214.060 | 215.710/216.545 | 187.285/187.642 | **143.351/142.813** |

Best tested: direct/two buffers/StageK128, **347.360/346.240 us**,
**772.787/775.287 GFLOP/s** for this **one CTA**, not device-wide TFLOP/s.
It is still far from the ~64-cycle MFMA-only rate. No unique limiting unit,
HBM traffic count, variance-controlled global optimum or full-GPU performance
claim. Double-reg's lack of speedup reflects the non-overlapping control;
do not attribute the entire gap solely to bypassing registers.

The first per-vector/per-fragment producer version is archived under
`temp/fp16-gemm64/serial-producer/`; only the batched versions remain active.

## Build / run / artifacts

```sh
# Depth, unroll, StageK, optional output directory.
# Depth1..8; StageK32/64/128; U must divide StageK/8. D and U need not divide each other.
bash research/fp16-gemm64/build.sh 2 4 128
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/d2-u4-t128
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/d2-u4-t128/probe \
  --mode gemm --producer direct --buffers 2 --k 32768 --rounds 9

# LDS-only two-by-two sharing diagnostic:
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/d2-u4-t128/probe \
  --mode lds-sweep --records 32 --steps 4096
python3 research/fp16-gemm64/run.py
```

The full runner expects D2 StageK32/64/128 and D1/D3/D4 StageK64 builds, U4.
Use `--resume` only without changing the experiment. Raw commands, copy checks,
CSV/logs, absolute wave clocks, snapshots, flags/hashes and inventories are in
`results/final/`. Initial sharing checks are under `results/grid/`. Binaries,
headers, code objects and extracted core intervals are under each `temp/`
build directory. A generated compile database supports clangd; use it rather
than the workspace's unrelated default CMake database. No production, CMake,
model fixtures or other research branches are modified.
