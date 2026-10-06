# AMD GPU notes for agents (gfx90a / CDNA2)

Status: living reference. Purpose: AMD GPUs are uncommon, and the same
mistakes recur across agents (this file exists because one did). Read
before reasoning about occupancy, registers, or `__launch_bounds__` on
this target. The engine targets **gfx90a** (MI250X, CDNA2, ROCm 7.1,
hipcc/clang 21). CUDA intuition is a liability here — several semantics
diverge, and three of them have already bitten this project.

Authoritative sources used below: ROCm optimization guide ("Register
pressure in AMD CDNA2 GPUs"), the CDNA2 dense MFMA builtins reference,
the LLVM AMDGPU backend docs / commit `ca0c92d`, and PR llvm/llvm-project
#85860. The empirical numbers are from this tree's own compiled kernels
(read via `--save-temps` + the `.AMDGPU.csdata` block, see §6).

## 1. VGPR and AGPR are separate register classes, not a unified file

This is the mistake that started this file. **Architectural VGPRs
(`archVGPR`) and accumulator VGPRs (`accVGPR` / AGPR) are distinct
register classes.** They are NOT interchangeable:

- Ordinary VALU instructions (the workhorse FMA/load/int/log units) can
  name **archVGPRs only** — never an AGPR as source or destination.
- MFMA instructions carry their **C/D (accumulator) operands in AGPRs**.
  On CDNA2 the A/B operands *may* be in either class, but the compiler
  chooses per operand; it is not free mixing. (CDNA1 forced A/B into
  VGPR and C/D into AGPR; CDNA2 relaxed A/B only.)
- Data moves between the two classes via **dedicated move instructions**
  (`v_accvgpr_read` / `v_accvgpr_write`). There is no implicit path.

What *is* shared is the **physical storage**: one 512-entry physical
register file per work-item, partitioned by the `ACCUM_OFFSET` field in
`COMPUTE_PGM_RSRC3_GFX90A` into `≤256 archVGPR + ≤256 accVGPR` (ISA
§3.6.4; PR #85860: "a total of 512 registers, but a maximum of only 256
of each of AGPR and VGPR"). That is physical sharing of storage, not
architectural interchangeability.

**Consequence agents keep missing:** a kernel with **no MFMA uses zero
AGPRs** (`NumAgprs: 0` in the resource report) and, per LLVM `ca0c92d`
("Allow to use a whole register file on gfx90a for VGPRs"), gets the
**full archVGPR budget** (up to 256 addressable) — no accumulator is
"reserved away" from it. So "an AGPR/accumulator reservation stole my
VGPR budget" is **not** a valid explanation for an occupancy collapse in
a pure-SIMT kernel. Every kernel in this engine's `gpu_rocm/kernels.hip`
is MFMA-free (`NumAgprs: 0`); if you see one spill or under-occupy, look
at VGPR count and LDS, not AGPRs.

The `NumVGPRsForWavesPerEU` / `AccumOffset` fields in the disassembly are
compiler codegen artifacts used for the occupancy calculation; they are
**not** the kernel's VGPR count (read `; NumVgprs:` / `; TotalNumVgprs:`
for that) and a nonzero `AccumOffset` on an AGPR-free kernel is not, by
itself, a bug.

## 2. Occupancy units: waves per SIMD, not waves per CU

The second recurring mistake. Occupancy figures on AMD are reported
**per SIMD (per EU)**, not per CU. A CDNA2 CU has **4 SIMDs**, each can
run **up to 8 wavefronts**, so the physical ceiling is **8 wavefronts /
SIMD = 32 wavefronts / CU**.

The `; Occupancy:` line in the `.AMDGPU.csdata` block (and the
`-Rpass-analysis=kernel-resource-usage` "Occupancy" remark) is **waves
per SIMD**. Proof from this tree: `k_prefill_combine` (12 VGPR, no LDS)
reports `; Occupancy: 8` — that is the per-SIMD max (it would be 32 if
the field were per-CU).

So `; Occupancy: 1` means **1 wavefront / SIMD = 4 wavefronts / CU**, not
"1 wavefront per CU." Confusing the two flips the conclusion by 4×. The
`-Wpass-failed` warning is equally per-SIMD: *"failed to meet occupancy
target given by 'amdgpu-waves-per-eu' … desired occupancy was 4, final
occupancy is 1"* — `waves-per-eu` = per SIMD.

## 3. `__launch_bounds__` on HIP: the 2nd arg is waves per SIMD

`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` is a
CUDA spelling, and on NVIDIA the second argument is blocks per SM. **On
HIP/AMDGPU the compiler maps it to `amdgpu-waves-per-eu` = waves per
SIMD (EU), not per CU.** This is a documented AMD divergence and the
root of the bug fixed in `k_prefill_split` (§5).

Compounding the trap: a **gfx9 wavefront is 64 threads**, so a 64-thread
block is exactly **one wavefront**. Therefore:

  `N blocks/CU  ⇔  N/4 wavefronts/SIMD`   (when block = 1 wavefront)

Examples on this device:
- `__launch_bounds__(64, 1)` → requests 1 wavefront/SIMD = **4 wavefronts/CU = 4 blocks/CU**.
- `__launch_bounds__(64, 4)` → requests 4 wavefronts/SIMD = **16 wavefronts/CU = 16 blocks/CU**.
- `__launch_bounds__(256, 1)` → 256 threads = 4 wavefronts/block; 1 wavefront/SIMD request caps a CU at 1 block (= 4 wavefronts). **Block-thread-count and waves-per-SIMD are independent axes** — do not conflate them.

If the compiler cannot meet the requested waves-per-SIMD (VGPR/LDS/SGPR
won't fit), it emits `-Wpass-failed` and falls back to the achievable
value. The request is a *hint*; it does not force occupancy.

## 4. The occupancy limiters (and how to compute them)

For an MFMA-free kernel the achievable waves/SIMD is the min of:

- **VGPR**: `floor(per-SIMD-VGPR-slots / allocated-VGPRs-per-wave)`, with
  allocation granularity (VGPRs are allocated in blocks; 116 used may
  allocate as 120). The per-SIMD file on MI250X is 128 KiB = 512 VGPR
  slots. (Rough rule from the ROCm blog's table: ≤96 VGPR → 5/SIMD,
  ≤102 → 4/SIMD, … dropping as VGPRs grow; ≤256 → ≥1/SIMD.)
- **LDS**: `floor(per-CU-LDS / LDS-per-block)`, then `/4` for per-SIMD.
  A 16 KB `__shared__` tile is a common limiter here.
- **SGPR**: rarely binding (`floor(800 / SGPRs-per-wave)`).
- **Hard cap**: 8 wavefronts/SIMD.

The binding limiter can differ by head dimension: in this engine
`k_prefill_split<64>` (116 VGPR, 16 KB LDS) is **LDS-limited** to 4
blocks/CU = 1 wave/SIMD, while `k_prefill_split<128>` (183 VGPR, 16 KB
LDS) is VGPR-capped at the same 1 wave/SIMD. Both land at 4 blocks/CU —
the design target — for different reasons. Always read the resource
report; do not reason from one limiter alone.

## 5. Case study: the `k_prefill_split` `__launch_bounds__` episode

The v4 split-T prefill kernel (`src/blocks/gpu_rocm/kernels.hip`,
ported from `research/prefill-attn-ab`) was annotated
`__launch_bounds__(64, 4)`. The author intended "4 blocks/CU" (the
design occupancy, an LDS-driven 4). On HIP, however, `(64, 4)` requests
**4 wavefronts/SIMD = 16 blocks/CU** — unachievable given the 16 KB LDS
tile (caps at 4 blocks/CU = 1 wave/SIMD). The compiler emitted
`-Wpass-failed` on every build and landed at `Occupancy: 1` (wave/SIMD)
= 4 blocks/CU — i.e. **the design target was met anyway, but the
annotation was wrong and the warning was spurious.**

Two agents then misread this as a real occupancy collapse:

1. One agent (and the original report that prompted this file) read
   `; Occupancy: 1` as "1 wavefront/CU" and concluded the kernel was
   running at ¼ of its intended occupancy. **Wrong unit** (§2): it is
   1/SIMD = 4/CU = the intended 4 blocks/CU.
2. Another agent explained the (nonexistent) collapse as "the AGPR/
   accumulator reservation steals the VGPR budget." **Wrong mechanism**
   (§1): the kernel has `NumAgprs: 0`, gets the full VGPR budget, and
   nothing spills (`ScratchSize: 0`).

**Fix applied:** `__launch_bounds__(64, 4)` → `__launch_bounds__(64, 1)`
on `k_prefill_split`, so the annotation states the truth (1 wavefront/
SIMD = 4 blocks/CU). Resource report unchanged (NumVgprs 116/183,
NumAgprs 0, ScratchSize 0, Occupancy 1 wave/SIMD); the `-Wpass-failed`
warning is gone; prefill throughput is unchanged (A/B on a 522-token
prompt: `(64,1)` ≈ 6200 tok/s vs `(64,4)` ≈ 6270 tok/s, within run-to-run
noise). The canonical equivalence gate (1B/8B teacher-forced + long
cases) stays green.

The research harness `research/prefill-attn-ab/ab.hip` keeps `(64, 4)` on
its `k_v4` to preserve the recorded A/B numbers; its comment now points
here. (Its `k_v4r` register-scores variant genuinely achieves
`Occupancy: 4` waves/SIMD, so its `(64, 4)` is correct and honored —
`k_v4r` is a different kernel that was never integrated.)

## 6. How to inspect a kernel's resources

Two equivalent ways; both read the same compiler data.

```sh
# (a) Compile-time remark (prints SGPRs/VGPRs/AGPRs/ScratchSize/Occupancy/LDS):
hipcc -O3 --offload-arch=gfx90a -Rpass-analysis=kernel-resource-usage \
      -x hip -c kernels.hip -o /dev/null

# (b) Assembly + the .AMDGPU.csdata block (also gives NumVGPRsForWavesPerEU,
#     AccumOffset, COMPUTE_PGM_RSRC fields):
hipcc -O3 --offload-arch=gfx90a -x hip -S kernels.hip -o kernels.s
grep -A6 'k_prefill_split' kernels.s | grep -E 'NumVgprs|NumAgprs|ScratchSize|LDSByteSize|Occupancy'
```

The fields that matter:
- `; NumVgprs:` / `; TotalNumVgprs:` — actual VGPR count (use this).
- `; NumAgprs:` — AGPR count. `0` ⇒ MFMA-free ⇒ full VGPR budget (§1).
- `; ScratchSize:` — spill memory (bytes/lane). `0` ⇒ no spill.
- `; LDSByteSize:` — `__shared__` per block.
- `; Occupancy:` — **waves per SIMD** (§2). × 4 = wavefronts/CU.
- `NumVGPRsForWavesPerEU` / `AccumOffset` — codegen artifacts for the
  occupancy calc; not the VGPR count, and not (by themselves) a bug.

The engine's exact compile command is in `build/compile_commands.json`
(grep for the `kernels.hip` entry).

## 7. Agent checklist (mistakes that have already happened once)

- [ ] "AGPR reservation stole the VGPR budget" — only possible in an
      MFMA-using kernel. Check `NumAgprs:` first; if `0`, AGPRs are
      irrelevant to your occupancy story.
- [ ] Reading `; Occupancy: N` as per-CU — it is per SIMD. ×4 for per-CU.
- [ ] Treating `__launch_bounds__(., N)`'s N as "blocks per CU" (CUDA
      style) — on HIP it is wavefronts per SIMD. 1 wave/SIMD = 4/CU.
- [ ] Conflating "blocks" and "wavefronts" — a 64-thread block is 1
      wavefront on gfx9; a 256-thread block is 4. State both when
      reasoning.
- [ ] Citing `NumVGPRsForWavesPerEU` as the kernel's VGPR count — read
      `; NumVgprs:` instead.
- [ ] Trusting a non-measured occupancy claim in prose — AGENTS.md and
      design §15 require every dispatch/occupancy clause to cite a
      measurement. If a doc asserts "N blocks/CU," find the resource
      report that backs it (or correct the doc).
- [ ] "The CPU reference / a CPU fallback will save us" — there is none;
      the engine is GPU-mandatory (design §3). A no-match device fails
      loud, never silently downgrades.

## 8. Pointers

- Design context: `docs/engine-design.md` §3 (execution model), §10
  (numerics/determinism/threading), §15 (kernel selection framework —
  cites the historical `__launch_bounds__` spill episode as the reason
  the measured-decision cache keys on compiler version; the specific
  spill is now resolved per §5 above).
- The kernel: `src/blocks/gpu_rocm/kernels.hip` (`k_prefill_split`).
- The A/B harness: `research/prefill-attn-ab/` (`ab.hip`, README.md).
- External: ROCm "Register pressure in CDNA2 GPUs" blog; CDNA2 dense
  MFMA builtins reference; LLVM AMDGPU backend docs; LLVM commit
  `ca0c92d`; PR llvm/llvm-project #85860.
