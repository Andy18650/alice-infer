# Alice Inference Engine — Design

Status: living document. GPU-resident execution is the only path; CPU
bring-up is retired. AGENTS.md tracks the machine and state of play.
Engine owns generation, host sampling, tokenizer metadata, and one worker /
Model per tensor-parallel rank. Factories shard weights; TensorView supports
strided host shards. Model state and the FP32 residual stream stay on device.

**TP collective integration (2026-10, user-directed, §17):** every block
publishes an immutable slot in a rank-local contribution slab and reduces
out-of-place into trunk io. S=1 uses native device-flag P2P: fused single-CTA
for <=16 KiB rows, pointer-hoisted/dephased data-major 2×1024 gather plus
ready barrier for the 32 KiB 70B row. S>1 uses out-of-place RCCL prefill.
There is no consumed epoch or copy-back. All workers complete GPU execution
before the next forward reuses slots. P2P epochs remain monotonic across
RCCL prefill gaps, graph replay, and reset. No rocSHMEM dependency.

Canonical's RCCL build is still broken. CMake selects the existing validated
AMD 7.1 overlay through `ALICE_RCCL_LIBRARY`; executable build RUNPATH and a
runtime symbol-path guard prevent accidental use of the wrong build. Engine
creates one load-phase communicator ID; workers initialize their own rank's
communicator after all models load, then record the S=1 compute graph.
`WorkItem` contains only execution commands/tokens, never initialization
payloads. Prefill remains eager; graph replay and rank threads are retained.

Named buffers are fixed-size inter-layer communication state, not loading
workspace. Their table is their sole owner; no grow API or duplicate ledger
registration. ROCm weight constructors own scoped device repack staging and
release it after completion during loading. All runtime work names the rank
execution stream; temporary upload/repack reuse is explicitly ordered.

Research remains standalone. Detailed cooperative, geometry, traversal,
lookahead, register-tile, and vendor results plus limitations are recorded in
`research/p2p-allreduce/README.md` and `findings.md` (research commit ba8eb92).
Only the selected simple decode gather and vendor prefill are integrated;
other variants stay research candidates. Cross-implementation numerics are
tolerance-based; canonical gates and end-to-end measurements are in §17.
A standalone persistent counter probe now measures increment/wait progress
without payload or per-step launches (`research/p2p-allreduce/counter.md`).
At ws=8, parallel all-peer polling is ~1.81/2.07 us per cycle for relaxed /
release increments versus serial ~5.58/5.65; neighbor-only readiness is
~1.00/1.16. These are native O3, CPU+memory NUMA node 0, no attention context,
10,000 timed steps after 64 warm steps, nine samples. They are synchronization
skeleton costs, not one-way propagation or a claimed ring-AllReduce floor.
No production synchronization changes were made. A separate single-GCD
entry-timestamp study (`research/kernel-launch/`) records raw core and steady
clocks per wave with no entry barrier. Many CTAs begin close together below
residency, but zero-hold bodies can finish before later CTAs enter; 1664 CTAs
with a 100 us hold show ~102 us admission spread versus a kernel-specific
832-CTA capacity. This describes GPU scheduling, not a CPU per-CTA setup
loop or an exact hardware-allocation timestamp. At 256 one-CTA markers,
eager/graph start gaps are ~2.24/1.28 us (native O3, GCD 0, CPU+memory NUMA
node 0, no attention context). The extended identical-kernel cooperative
control is slower (~19.36 us start gaps versus normal graph ~1.60 for 256
one-CTA markers). Direct oversize cooperative launch rejects at the measured
832-CTA capacity, but captured replay accepts 1664 with device cooperative
context valid. The probe records this API inconsistency as unexpected, with
no grid.sync() and no valid timing claim; applications must enforce residency
before cooperative grid synchronization. No production launch changes made.

A standalone local VRAM ceiling sweep (`research/vram-ceiling/`) now exceeds
that earlier decode study's 1290 GB/s reference. Native O3 HIP/C++20,
HIP logical device 0 (PCI 0000:31:00.0, 104 CUs), CPU+memory NUMA node 0,
no model/context: 576 checked configurations vary load width, independent
chains, grid/block geometry and traversal. Two final process runs confirm
1 GiB best medians 1530.5/1509.3 GB/s; the same 8-byte-load configuration
sustains 1462.8/1460.2 GB/s medians at 4 GiB across nine randomized rounds.
Actual-code-object disassembly preserves wide loads; sampled active clocks
are 1700/1600 MHz core/memory without setting overrides. Treat ~1.46 TB/s
as a provisional multi-GiB local-read reference, not a GEMV/all-GCD ceiling;
size-dependent ranking and timing variance remain. An exact-code-object
occupancy query finds 59 VGPRs, no LDS/scratch, and 8×256-thread CTAs/CU
(832 across 104 CUs). A same-occupancy timestamped twin confirms all 3328
waves overlap at the 832-CTA launch, whereas a 1664-CTA control does not
fit the whole grid simultaneously. This is observed reproduction plus
capacity evidence, not a historical hardware-counter trace. No production
changes.

The follow-up `research/vram-addressing/` probes fixed-payload tile strides,
base alignment and disjoint paired streams on that same GCD/placement,
without a model/context. Native O3, 256 B tiles, 32 MiB useful reads:
8 KiB stride is ~1100 GB/s; 16/32/64 KiB fall to ~889/557/304, while +256 B
padding restores ~1105/1085/1070. Seven randomized rounds and independent
32/64 MiB process repeats retain the losses. Paired streams show periodic
half-stride relief and a reproducible difference between 8/16/24 GiB plane
gaps. These are effective contention signatures, not proof of a 256 B
physical HBM interleave or an XOR swizzle. Counter collection is blocked by
AQL-profile extension initialization; no system workaround or production
change was made. See the study for codegen, checksum and mapping caveats.

`research/vram-ilp/` now separates addressing from compiler scheduling at
104×256 threads on that same GCD/NUMA placement, native O3, no model/context.
The preserved U=16 global reader emits load→vmcnt(0)→consume repeatedly;
U=8 batches loads. A scheduling-constrained U=16 reader keeps addresses
unchanged and restores ~559→1396 GB/s at 1 GiB (nine rounds of 12 reads).
Wave-local 4 KiB tiles reach repeated ~1480 GB/s medians at 1 GiB and
~1420 at 4 GiB; one 1502.9 median repeats at 1486.4, not a stable 1500+
claim. A timestamped twin places one CTA on each of 104 CUs with four
waves/CU. Larger unrolls, launch bounds and ping-pong banks are not uniform
wins. Matched unchanged 832-CTA controls run ~1400 here, below their earlier
1530.5; that cross-campaign shift is not isolated. No production changes.

`research/vram-coalescing/` compares equal 256-dword wave tiles using A:
per-lane 16 B vector loads, B: four adjacent-lane scalar passes, and C:
four per-lane contiguous scalar loads. Same GCD/NUMA placement, native O3,
no model/context, 1 GiB, 104×256, four independent tiles: A/B repeat at
~1430 GB/s, controlled scalar C ~676. B/C have identical scalar load counts,
graduated waits, resource capacity and cache flags; more batching or 832
CTAs does not rescue C. Normal LLVM optimization widens C into A's 16 B
loads and restores ~1430. This isolates instruction/address-pattern
sensitivity, not physical HBM traffic or a channel map. A same-capacity
hardware-ID/timestamp follow-up (`research/vram-coalescing/packing.md`)
finds identical CU/SIMD spreading for A/B/C: 104/208/416/832 CTAs give
1/2/4/8 overlapping CTAs per CU, each CTA's four waves split one per SIMD.
A 1664-CTA control peaks at 3328 waves, with later assignment as work
finishes. Two process runs × seven samples confirm this workload's balanced
packing, not a general memory-contention-minimization policy. The current
probe now uses raw uint32_t input and HIP uint4, with no float bitcasts or
custom vector aliases. Revalidation retains the A/B ~1430 versus controlled
C ~675 GB/s pattern and the same CU/SIMD packing; historical float-bit
snapshots remain separately labeled. A uint4 expression can split into two
64-bit loads in the controlled build, which the ISA checker records rather
than treating the source type as an instruction guarantee. Barrier removal
is now the default research path, with separate compiler-barrier controls:
A/chunks=8 and optimized C/chunks=4 return to serial load/wait/use and slow
to ~1020 GB/s, while A/B at four chunks retain batching. The narrowed
barrier-free A/B four-chunk test repeats at grids 104/208/416: 1 GiB A is
~1435/1355/1309 GB/s, B ~1444/1417/1371; more blocks do not improve the
winner. These are native O3, same GCD/NUMA placement, no model/context,
two process runs of nine rounds; grid count also changes address spacing.
See `research/vram-coalescing/tuning.md`. A rotating register-buffer
follow-up (`ring.md`) finds that unconstrained C++ is rebatch-scheduled;
per-slot compiler constraints retain steady waits at vmcnt(3) for A and
15..12 for B, with no steady-state drain to zero. Ordered B depth 4 at
104×256 repeats at 1470.3/1465.6 GB/s versus batch B 1426.6/1434.9 at
1 GiB; at 4 GiB 1423.7/1424.0 versus 1401.8/1398.9, winning all paired
rounds. A does not improve. The exact historical uint2/U16 code object,
interleaved on the same input, still runs ~1400 rather than 1530. These
are native O3, same GPU/NUMA placement, no model/context. The small B gain
retains the experiment rather than triggering the requested failure/revert
path; the original source remains unchanged for inspection. No production changes.

## 1. Purpose & target

A from-scratch C++20 LLM inference engine for this machine: 4× MI250X
(8 GCDs × 64 GB HBM2e, gfx90a, ROCm 7.1) behind a 48-core Zen3 host.
Relatively high-performance and relatively general: built for this
hardware first, structured so other GPU vendors can be added later
without rearchitecting.

Code quality is a feature, not a preference. Development is agent-heavy,
and ill-shaped code compounds under agent edits into "playdough": layers
of patches and one-time constraints squeezing the project out of shape.
The codebase must stay small, readable, and self-documenting; the tech
stack (CMake, C++, vendor libraries, hipcc) is chosen by what produces
good code and good results, never the reverse, and no decision is kept
past the moment it is demonstrably wrong.

Bring-up models: Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-Distill-Llama-70B (HuggingFace sources under `/opt/models`,
packed into `/opt/models/alice/<name>/` by `tools/pack_model.py`). The 70B
(141 GB bf16, 80 layers, 17 shards) is the sizing reference.

## 2. Deployment assumptions

Single user. One live context at a time; every new input is appended to
*the* context. No prefix matching, no rollback requests at the engine
level. Consequences: no multi-context registry or locking; stateful
Blocks own their state directly, sized once for max context; state ops
reduce to write-during-forward and `reset_state()`; memory planning is
fully static at load. If multi-session ever arrives it is bolted on by
instantiating state per session — compute never rearchitects, because
state lives behind Block boundaries.

## 3. Execution model

The engine requires a GPU. There are no CPU-only builds.

All model state lives on the device: the residual stream, the block io
buffer, attention KV, norm weights, RoPE tables. All state operations —
normalization, residual adds, position encoding — execute on the device.
A forward pass performs no host↔device transfers inside the block loop;
the regular traffic is token ids (in at the embedder, out at the sampler
— four bytes per step). The host loads weights, tokenizes, detokenizes,
and runs the frontend.

The stream is FP32 end to end. The numerics boundary that was paid for
during bring-up is kept — only GEMM operands are quantized (bf16
storage, fp32 accumulate). What changed is the stream's address space,
not its precision.

Blocks are written for their target architecture, in that architecture's
native language and libraries. There is no vendor-compatibility shim
inside Blocks, because optimized-kernel reality does not abstract:
FlashAttention has a different best implementation per SM generation,
wgmma source does not compile across adjacent NVIDIA architectures,
NVFP4 is native on sm_100 and emulated elsewhere, and the right tensor-
parallel strategy depends on link topology. A shim would leak at exactly
the points that matter. Portability is instead achieved by writing
several native implementations of one abstract block class and selecting
at load the one matching the device present. Vendor BLAS is used where
it genuinely covers the problem (dense GEMM) and not where it doesn't
(attention, fused elementwise chains, quantized formats). Measured
addition (2026-09, research/decode-roofline): S=1 GEMV is a "doesn't"
— rocBLAS gemm_ex picked no-split-K kernels running the FFN shapes at
313-727 GB/s against a 1290 GB/s read ceiling (26 of 29.6 ms/token).
The gpu_rocm family now carries its own split-K GEMV, dispatched at
S==1 inside both blocks; see §15.

The Trunk's own operations are the deliberate exception: norms, residual
adds, and RoPE are plain SIMT kernels — no matrix intrinsics, no warp
tricks — written once against a small macro header that maps kernel
spelling onto HIP/CUDA/MUSA-flavored compilers. That header exists for
this level only and never crosses a Block boundary.

CPU Blocks survive in exactly one role: development reference. A CPU
reference block implements the same Block interface by copying its input
to the host, computing with naive, maximally readable code, and copying
the result back. Its purpose is fault localization in research-grade
kernels: a correct Flash Linear Attention implementation is a research
project of its own, and comparing a full GPU model against a full CPU
model cannot isolate which block diverges — swapping one block instance
at a time can. Transfer cost is irrelevant because such blocks never run
in production. One practical exception may eventually appear — CPU-
resident MoE experts — and will be designed for if it does.

Device code is compiled with the vendor toolchain (hipcc on this
machine). Runtime-compiled string kernels (hipRTC) are retired: they
existed to keep hipcc out of the build, and the build no longer avoids
hipcc.

## 4. Model = program, .model descriptor = contract

The engine loads via the alice model format: a packed `.model` file
(one JSON object + `\n\0`) + a sibling `weights.bin`. The JSON's
`model` map is the **descriptor** — composition (block specs, members,
blocks), each carrying only a content-ID (the BLAKE3 of its bytes); the
unsigned `files` array maps IDs to `{file, offset, size, codec}`
locations. The descriptor is the *total contract*.

Identity is separated from location: a block/member record carries only
its content-ID; the ID doubles as the integrity check (the DataLoader
verifies `BLAKE3(bytes) == id` on every fetch, so no caller ever sees
unverified bytes). Content-addressing dedups — identical bytes share one
ID and one stored copy (tied weights: the lm_head names the embedder's
ID, the one legal shared claim, fetched twice but stored once). The
factory's `derived_bytes()` is the two-watches check: it derives the
expected record size from the spec and reconciles it against the fetched
blob in `construct()`.

Sharding is physical packaging, invisible to the contract: the packer
(`tools/pack_model.py`) consolidates HuggingFace shards into one
`weights.bin`; no HF `index.json` is read at load time. The packer is the
only producer of `.model` files.

## 5. Tensor & dtype contract

The typed tensor `Tensor<T>` is the **host-side vocabulary** (2026-09
revision: the former non-owning `TView<T>` and the loader's `Staging`
merged into one move-only owning type — exactly one owner per buffer,
one live handle, "freed on return" is mechanical rather than a caller
convention). Owning, rank ≤ 2, row-major, PACKED — deliberately not
strided: transposes are explicit decisions at GEMM call sites (rocBLAS
op flags), never metadata. The strong `bf16` struct. Runtime `DType`
survives only in file-facing structs; the descriptor's block specs are
dtype-free, the expected dtype being derived from the block type and
enforced at the factory's `construct()` (the load-boundary check is the
DataLoader's BLAKE3 verify, not a dtype check). Quantized formats get
concrete composite types (data + scales), never a polymorphic hierarchy.

The device-side stream and device buffers are not Tensors. They are raw
device allocations owned by components, referenced as plain pointers in
device code; borrowed host access at compute boundaries (oracle kernels,
the host Sampler) is likewise ptr + dims, mirroring the `gpu::Block` io
contract.

## 6. Component taxonomy

- **Functional / Weightful / Stateful / Block** — as before: stateless
  primitives; weight owners; state owners (KV, SSM state); the Block is
  the smallest unit the descriptor describes.
- **Embedder** — token ids → stream rows (a device-side gather).
- **LmHead** — final hidden → logits (device GEMM; last row only at
  prefill). Deliberately not bent into the Block signature.
- **Sampler** — logits → token id, on device; owns RNG state. The
  sampled id is the only per-step D2H unless a frontend asks for logits.
- **Trunk** — owns the device-resident stream buffers, ALL norm weights,
  the norm/residual SIMT kernels, and the wiring scheme. A model with a
  new residual scheme gets a new Trunk, by design.
- **Model** — container + wiring loop; owns parts via `unique_ptr`;
  knows no architecture specifics. Token ids in → token id out.

## 7. Forward loop

```
io = embedder.embed(token_ids)          // gather into the fp32 stream
for block in blocks:
    contribution = slot(i) if TP > 1 else io
    trunk.read(i, contribution)          // residual → contribution, apply norm
    block.forward(contribution)         // in place; publish immutable TP slot
    if TP > 1: allreduce(slot(i), io)    // produced barrier, gather → local io
    trunk.write(i)                      // residual += io
token = sampler.sample(lm_head.project(trunk.output()))   // on device
```

Prefill processes the entire prompt including the last token. Invariant:
every token enters Block state exactly once, and its logits are
available immediately after processing. Position never crosses any
interface; each stateful Block keeps a private position counter and may
evict/compact its state at will — strictly its own business.

## 8. Class structure

### 8.1 Blocks

- `Block`: `forward` over the shared device stream, in place; virtual
  destructor. `StatefulBlock` adds `reset_state()` plus default-trap
  snapshot methods (future session API).
- One abstract class per (block type × weight dtype), unchanged: dtype
  dispatch stays at the type level, so no implementation carries
  per-dtype switches with per-architecture ifdefs.
- Concrete implementations are `final`, one per target architecture
  (`DenseFfnBF16Gfx90a`, …). A factory selects the implementation for
  the device present. **There is no fallback chain terminating in a CPU
  impl**: if nothing matches the present hardware, loading fails loudly.
  Silent downgrade was a bring-up convenience; in production it hides
  configuration errors.
- CPU reference blocks (retired): one naive host implementation per
  abstract block class, swappable in for development (§3), was the bring-
  up oracle. Self-contained — the D2H/H2D copies lived inside the
  reference block, so the Block interface stayed uniform and nothing
  else knew they existed. Retired 2026-09: naive reference impls for new
  block types are now done as GPU kernels, not host blocks; the host-side
  math helpers survive as `src/host_math`. The Block interface stays
  uniform; nothing else knows the impl exists.
- Self-containment, unchanged: private weight storage, private math, no
  shared kernel toolbox across impls. Constructors take already
  validated staged weights (`Tensor<T>`, moved in), copy/convert/pack
  into private layout (staging H2D included); the staging dies inside
  the constructor, so "freed on return" is mechanical. Contract
  amendment (2026-09): few-role blocks take single staged tensors
  positionally; the trunk — the one component whose weights are a
  homogeneous set — takes the staging vector itself.

### 8.2 Trunk

Role unchanged (the "where the mess lives" class), now device-resident:
owns the stream buffers, norm weights, and the plain-SIMT norm/residual
kernels of §3. First implementation: RMSNorm + add-residual. Revisit
triggers for merging norm into Trunk: trunk-impl proliferation,
mixed-norm models.

### 8.3 Model, Embedder, LmHead, Sampler

Thin concrete classes; no dtype×arch abstract matrix until a second
implementation exists. Sampler config fixed at construction (temperature
0 = argmax for tests); the Sampler owns the RNG — acceptable mutable
state under single-session.

Model's shape (2026-09): named members over forward-declared incomplete
types — the "exploded pimpl" (one Impl indirection was tried and
rejected; `private:` leaking layout is solved by not including the
component headers, not by hiding them behind a pointer). The public
header stays free of internal- and vendor-includes (no trunk.hpp, no
HIP/rocBLAS); their headers are included only by the implementation
TU. Consequence, accepted deliberately: a change to the component SET
edits the public header; a change inside a component never does. The
GPU Context (device binding + rocBLAS handle) is owned by the Model,
like the shared scratch — session-scoped GPU resources live under
Model, never in process-wide state.

### 8.4 ModelLoader

Narrow entry that sniffs format by content and delegates
(`SafetensorsLoader` now); registry machinery deferred until format #3.

## 9. Memory architecture & weight loading

- No allocation during `forward`; everything is sized at load.
- Private buffers per component — device allocations now: weights,
  state, transients.
- Named buffers allow layers to communicate beyond the residual stream.
  Same name + size returns the same stable pointer; any size disagreement
  fails loudly. They never grow and live until model teardown. They are
  separate from the block arena and attention scratch, and are not a
  temporary-workspace allocator.
- Column-major weight repacking uses constructor-local device staging
  (`hipMalloc`, scoped `hipFree`), released after repack completion during
  loading, including on exception. No repack staging survives into inference.
- The shared stream buffers (residual, io) are device allocations owned
  by the Trunk, `[max_prefill_chunk, d]` fp32. Under TP, Model additionally
  owns a contribution slab `[n_blocks, max_prefill_chunk, d]` fp32 per rank
  (no slab at TP=1). Each block normalizes into and computes in its slot;
  once published, the slot is immutable through forward completion.
  Allreduce reads peer slab bases plus the same fixed slot offset and writes
  local trunk io, never a published slot. Slab allocation checks current
  free memory and reports its footprint / max_prefill_chunk on failure.
  One more cross-block sharing instance exists (2026-09, first §14 scratch-pool use): the v4
  prefill attention partials buffer, loader-allocated once
  (`nh × max_prefill_chunk × ceil(max_seq/64) × (hd+2)` fp32) and reused
  by every attention block — each block's partials are produced and
  consumed within its own `forward` (split → combine).
- Load flow, per block in order: `pread` shard bytes → pinned host
  staging → H2D → staging freed. Peak host RAM ≈ one block's staging.
  Pinned staging is mandatory (pageable H2D runs a driver-staged path,
  measured 2.2 vs 28.5 GB/s); the pinning API lives inside the GPU
  platform module, not in core code.
- **No mmap anywhere.** Weight streaming is one-time, not re-read; the
  page cache would duplicate data and add RSS/reclaim pathologies. If
  the SSD-resident regime ever arrives, the answer is an explicit
  user-space cache (the Colibri playbook), designed then.
- Multi-die: the 70B needs ≥3 GCDs. Blocks get a placement; the stream
  crosses between devices by peer-to-peer copy at placement boundaries.
  A named seam, built when 70B tuning demands it.

## 10. Numerics, determinism, threading

- Numerics policy (user-signed): cross-implementation equivalence is
  **tolerance-based**; within-impl determinism holds per (impl, S,
  kernel schedule). The teacher-forced canonical test vs HF is the
  equivalence gate.
- Device-side determinism: fixed kernel schedules; vendor BLAS invoked
  with fixed algorithms (no run-to-run autotuning).
- The model loop is strictly sequential; intra-op parallelism is each
  implementation's private business. A forward work item completes GPU
  execution before its worker reports done: logits paths use async D2H plus
  synchronization on the rank stream, and `ForwardRun` synchronizes that
  same stream. The Engine's
  all-worker completion barrier therefore protects contribution-slot reuse
  across forwards. No host synchronization is added inside the block loop.

## 11. Extensibility targets (designed for, not built)

- Quantized weight formats (MXFP4, NVFP4, FP8-B128): a format is a
  bundle of tensors plus compute semantics, landing as new values on the
  block-type × dtype axis — with per-arch native implementations (NVFP4
  native on sm_100, emulated elsewhere; exactly why Blocks carry no
  shim). The constructor contract already permits dequantize-at-load
  fallbacks; the two-way audit gains format-specific shape relations.
- New block types: MoE, linear/hybrid attention (KDA/FLA — expect the
  CPU-reference-block workflow of §3 to be the bring-up path), SSM.
- New trunk schemes: mHC (GLM), attention residuals (Kimi).
- Multi-die placement (§9). CUDA/HIP graph capture when decode
  approaches the kernel-launch-latency floor (~150–200 launches/token).

## 12. Non-goals

- CPU-only operation (abandoned 2026-09).
- A vendor-abstraction layer inside Blocks (§3).
- Tokenizer / chat template / generation loop inside `Model` (frontend).
- GGUF. Multi-request scheduling. Fork/beam/speculative state ops.
  Python in the engine core.

## 13. Testing strategy

- The equivalence gate: teacher-forced canonical logits vs HF per model
  (`tests/canonical*/`, gate max|Δlogit| < 0.75). Watch item: max drift
  grows with depth (0.22 / 0.28 / 0.68 at 16 / 32 / 80 layers).
- **Reference quality (user-signed 2026-09)**: short fixtures stay bf16-
  reference with the 0.75 max gate. Long fixtures (~1500-token WikiText
  prompts, regenerable via `dump_canonical.py --prompt-file`, gitignored)
  use an **fp32-ideal reference** (torch `dtype=float32` upcasts weights
  exactly and runs the stock graph in fp32 — the mathematical ideal of
  our fp32-stream design; measured: the CPU engine is EXACT against it,
  max 0.0002, while PyTorch's own bf16 inference sits 15.6/1.8 max away).
  Gates on the long case: GPU suite mean ≤ 0.05 AND max ≤ 0.75... (the
  bf16 GEMM-input budget at massive-activation positions; observed 0.018
  / 0.76 / 1.52). (A temporary host-path CPU long suite gated at 0.01
  proved the semantics exact — 0.0002 — and was retired with the host
  path; the proof stands recorded here.)
  Greedy decode comparison is exact-match with a coin-flip rule: a
  divergence is a bug only if our token is ≥0.25 below the reference
  argmax (near-ties flip legitimately within the bf16-input budget).
- **Single-block HF oracle (2026-09)**: `tests/python/dump_hf_block_oracle.py`
  extracts real weights from a chosen layer and runs the exact HF
  `LlamaAttention` / `LlamaMLP` submodules on synthetic data.  The C++
  test constructs the corresponding alice::gpu_rocm block with those
  weights and asserts against the HF output.  Because only one block is
  exercised (no layer-stack amplification), the bound is much tighter
  than the full-model gate: observed max|Δ| 0.040 prefill / 0.003 decode
  for attention, 0.016 prefill / 0.010 decode for FFN (Llama-3.2-1B
  layer 5, S=8/S=1).  This is the primary per-block numerical oracle.
- **CPU reference (§3)**: retired. The `cpu_generic` host-block reference
  impls are gone (naive reference impls for new block types are now done
  as GPU kernels, not host blocks); the `ALICE_CPU_REF_BLOCKS` dev
  fault-localization swap-in retired with them. The host-side math
  helpers (RoPE tables, rmsnorm/add_rows/silu — the kernel-level oracle
  for `test_gpu_kernels.cpp`) survive as `src/host_math.{hpp,cpp}`.
  `test_gpu_blocks.cpp` keeps only structural tests (incremental==
  batched, reset freshness) that have no HF equivalent.
- Determinism: same tokens + fresh state + temperature 0 → identical
  output.
- Loader: the DataLoader's BLAKE3-verify path (corruption = hash
  mismatch, fails loud); the real packed `.model` files as fixtures.
  Ad-hoc models validate with zero recompile via `CanonicalEnv`
  (`ALICE_CANONICAL_{MODEL_FILE,FIXTURES[,CHUNK]}`).
- Slab allreduce: four-rank kernel test covers separate output, unchanged
  published slots/padding, tiny fused / 32 KiB gather paths and tails, and successive
  forward epochs. TP=2 canonical validation covers wide prefill, S≤4 and S=1,
  reinitialization, and chunked `generate()` with `ForwardRun` completion.
- Tests require a GPU, as the engine does.

## 14. Open questions

- Prefill/decode interface separation beyond shape dispatch — circle
  back for SSM/linear-attention blocks (chunkwise-parallel prefill vs
  recurrent decode) and speculative decoding (short-prefill-with-
  rollbacks; likely a new interface).
- Device transient sharing / scratch pool (first instance landed: the
  shared v4 prefill partials buffer, loader-owned, §9; generalize under
  memory pressure or when a second sharer appears).
- Trunk/norm merge: revisit triggers in §8.2.
- Naming: LmHead ("de-embedder" candidate; `lm_head` matches files).

## 15. Kernel selection framework (2026-09, foundation of the next-gen loader)

A kernel choice is pinpointed by, in order of when the information
becomes available:

- **Block type**, including dtype and quantization strategy.
- **Model** (per-model shapes; 1B vs 70B can warrant different kernels).
- **Hardware SKU**, not just ISA: same-arch parts differ in CU count,
  clocks, L2; winners can flip between SKUs (MI210 vs MI250X).
- **Parallel configuration** (TP degree): cuts matrices differently and
  adds the collective itself as a selection point (hand-rolled P2P vs
  overlay-RCCL, keyed by degree and message-size class; measurements in
  research/rccl-single-process/).
- **Call-time shape vector**: today `(S, T)` with T collapsed by
  measurement (v2 wins at all T); batching or speculative decoding will
  add B. The schema carries the full vector even when clauses ignore
  parts of it.

Deliberately excluded: **data-dependent dispatch** (e.g. MoE routing
makes shapes value-dependent). The accepted mask-and-allreduce EP
scheme keeps the dispatch closed — fixed expert GEMMs, masked.

The selection pipeline narrows the candidate set at four times:

1. **Code time** (architecture design): cull provably-inferior
   implementations; a human act backed by harness evidence. Case study:
   v1 attention retired after measurement falsified both of its
   dispatch boundaries. Second case study (2026-09): at S==1 the
   rocBLAS gemm_ex path was replaced by a family-private split-K GEMV
   in gpu_rocm — measured 26.0 of 29.6 ms/token at 313–1143 GB/s per
   shape vs a 1290 GB/s streaming ceiling; the GEMV lands 933–1276 GB/s
   and the block dispatch clause is `S == 1 → GEMV, else gemm_ex`
   (decode-roofline study: replica harness linking the engine's real
   kernel objects, per-shape A/B, CTA-target sweep).
2. **Compile time**: per-arch implementation TUs. Deferrable to load
   time via fat binaries (`--offload-arch` / `-gencode` lists; the
   driver picks the code object) so one package can serve all
   platforms; the cost is the per-arch impl families we already accept.
3. **Load time**: model + hardware + TP + config decisively known.
   Narrow to a one-dimensional table over the call-time shape vector.
   The loader orchestrates (.model descriptor + EngineConfig + platform
   probe → key), constructors receive their table, `forward()` does an
   O(1)
   dispatch (ideal: one integer compare, e.g. `S > 4`).
4. **Call time**: index the table; nothing else.

Norms (both already emergent practice, codified here):

- **Every dispatch clause cites a measurement.** Unmeasured entries are
  allowed for bootstrapping but must be marked provisional.
- **No entry = loud failure**, never silent substitution. A table miss
  means "this configuration was never validated."

**Measured-decision cache.** The tables are filled empirically, not by
reasoning: key = (model, device SKU, TP, software versions — kernel
source + compiler + flags, since codegen shifts flip winners; the
`__launch_bounds__` spill episode is the evidence), value = the
narrowed table plus the benchmark numbers that produced it. First load
on a machine may run a seconds-long autotune pass over a *closed,
precompiled* candidate set (no JIT, no runtime SDK — our answer to
FlashInfer's precomputed cubin matrix); subsequent loads read the
cache.

**Capacity is a fit check, not a selection dimension** (user decision,
2026-09). EngineConfig values (`max_seq_len`, `max_prefill_chunk`)
never affect which kernel is right — they only size buffers, and a cap
on S never changes a clause over the S axis. Since all allocation is
load-time anyway, the loader computes the complete VRAM bill (weights +
trunks + KV caches + scratch + lm_head, linear in the config knobs)
before the first allocation and compares against free memory — per-GCD
under TP, with a margin — failing with a message that names the knobs
and the largest fitting `max_seq_len`. Explicitly NOT done: silently
swapping to a leaner kernel under memory pressure — that would be an
unmeasured silent regression; if a model cannot fit at any reasonable
config, a leaner dispatch entry is a deliberate user-signed decision,
not a runtime adaptation.

## 16. S=1 HIP graph replay (2026-10)

One graph per rank records the block loop, final norm, and language head;
token upload/embedding and logits download remain outside capture, ordered
on the same explicit rank stream. The language head is warmed separately
before capture (never warm a full rank while its peers are idle at an
allreduce). No allocations occur during replay. Graph creation failure is
loud, not an eager fallback. `ModelConfig::decode_graph=false` is the
explicit eager A/B baseline; the REPL exposes `ALICE_DECODE_GRAPH=0`.

Replay parameters are instance-owned device allocations, not process-wide
constant symbols. Each attention block has a private device position cursor:
prefill RoPE publishes the next position, decode reads the cursor, and the
output-projection combine increments it without an extra launch. A private
host mirror validates context limits / supports subsequent eager prefill;
`StatefulBlock::advance_decode_state()` advances that mirror before replay.
Recording executes host bookkeeping but not kernels, so initialization
restores cursors after capture. Reset clears both position mirrors, but never
resets collective epochs.

Decode attention has a fixed physical grid of 64 splits per local query
head. Kernels read the current device position to derive active splits and
rows per split, preserving the old logical partition: 64 rows/split through
T=4096, then ceil(T/64). Partial stride is always 64; inactive split blocks
return and combine excludes their unwritten entries. This avoids per-token
graph updates/recapture. The overlaunch cost at short contexts is provisional;
smaller-capacity graph variants remain a measured future optimization.

Captured native decode collectives use immutable per-block slab offsets and static
collective ordinals plus a device epoch base uploaded once before each
replay. Host epochs advance by the block count; eager prefill and graph
replay therefore share one monotonically increasing sequence. Flags are
never cleared per replay (an independently cleared peer could still expose
stale readiness). Only the first produced barrier remains in each collective.

Execution initially uses regular blocking streams for a conservative
migration. Load-time repack uploads and kernels nevertheless explicitly use
the same rank stream and synchronize before scoped staging release, fixing
the previous nonblocking-stream staging race rather than relying on implicit
ordering. Input/output copies and replay completion also name the rank stream.

Validation includes one captured attention graph replayed across T=64/128
and 4096 boundaries for hd=64/128 with NaN-poisoned inactive scratch, plus
TP=2 graph/eager logits comparison through changing tokens, prefill after
replay, reset, and context overflow. CanonicalEnv supports optional TP and
decode-graph config (`ALICE_CANONICAL_{TP,DECODE_GRAPH}`).

Initial A/B (Release, `numactl --cpunodebind=0 --membind=0`, 70B, 28-token
prompt + 128 decode steps, attention context T=29..156, max_seq=8192):

| TP | Eager decode | Graph decode |
|---|---|---|
| 4 | 45.22 ms/tok, 22.1 tok/s | 44.29 ms/tok, 22.6 tok/s |
| 8 | 36.15 ms/tok, 27.7 tok/s | 34.91 ms/tok, 28.6 tok/s |

One run per configuration, not a variance-controlled sweep; all four
128-token generated texts were identical. Logs are in
`temp/graph-decode-ab/` (not committed). 70B S=1 canonical replay passed at
TP=4 (worst max drift 0.4951) and TP=8 (0.5040), gate 0.75. Full Debug
suite: 35 passed + one optional CanonicalEnv skip; targeted Release replay
and long-context tests passed three consecutive runs. The fixed-64 grid
is a correctness-first baseline; overlaunch at short T can offset some
host-submission savings.

## 17. Hybrid TP collectives: integration and validation (2026-10)

User-directed integration after standalone research commit ba8eb92:

- S=1 / <=16 KiB: retain the existing fused single-CTA ready+gather kernel.
  The initial extrapolation of the larger row's two-launch strategy regressed
  1B decode; the small-row path is not silently replaced.
- S=1 / 32 KiB: produced flag barrier + pointer-hoisted/dephased data-major
  gather, 2×1024. Contiguous CTA chunks; 64 B LDS peer-address cache; no
  payload prefetch or register-heavy peer-major tiling in production.
- S>1: out-of-place `ncclAllReduce` from the immutable block slot to trunk io,
  using one rank-owned prefill communicator on the rank stream. No added
  per-block host synchronization. The all-worker forward completion boundary
  still protects source reuse.
- Host collective counters advance for RCCL as well as native calls; RCCL
  leaves P2P flags alone. Next replay uses a higher base, so gaps are safe.
  Reset clears attention state, never collective counters.
- CMake links `ALICE_RCCL_LIBRARY` (default: the documented existing AMD 7.1
  overlay), with executable build RUNPATH. A symbol-path guard rejects a
  different loaded library before worker/communicator initialization. Version
  equality cannot detect Canonical's known launch-path bug. UID lives only in
  the load-phase ThreadState; owner workers call `ncclCommInitRank` after load.
  Decoder graphs are destroyed before communicator teardown, with rank device
  binding restored for cleanup. No initialization WorkItem was reintroduced.

Dephased FP32 addition order can differ across ranks. It is not a bitwise
replica claim: every implementation must pass the signed numerical gates.
1B TP=2, 70B TP=4 and TP=8 pass committed canonical cases for eager prefill
(chunk=512) and S=1 graph replay (chunk=1). Worst 70B drift is 0.547702,
below 0.75; 1B max is 0.356657. Existing graph/eager, reset, overflow, and
chunked-generation tests cover switching between vendor/native collectives.
The runtime wrong-library guard is tested with an explicit system-library
LD override and fails safely before a collective.

### Matched engine A/B

Baseline ba8eb92 built separately with the same Release flags; new hybrid
built in build-release. Public Engine API driver loads once, validates all
three 70B fixtures in both chunk modes, then generates 128 tokens per round.
No streaming callback or per-token text printing. CPU and memory are bound
with `numactl --cpunodebind=0 --membind=0`; max_seq=8192, chunk=512.
Short prompt is 28 tokens (attention T=29..156), long prompt 1667 tokens
(T=1668..1795). Short table uses the second warm round; these are matched
initial runs, not a variance-controlled campaign.

| TP | short decode baseline → hybrid | long decode baseline → hybrid | 1667-token prefill baseline → hybrid |
|---|---|---|---|
| 4 | 44.72 → 44.18 ms/tok (22.36 → 22.63 tok/s) | 46.89 → 46.59 ms/tok (21.33 → 21.46 tok/s) | 1.853 → 1.449 s (900 → 1150 tok/s) |
| 8 | 34.99 → 33.20 ms/tok (28.58 → 30.12 tok/s) | 36.76 → 34.92 ms/tok (27.20 → 28.64 tok/s) | 2.244 → 0.917 s (743 → 1818 tok/s) |

TP=8 long prefill improves ~2.45×. All short generated texts match the old
binary; long TP=4 matches, long TP=8 differs. The committed teacher-forced
canonical gate passes; no fp32-ideal long-70B greedy-margin fixture was used,
so do not claim exact long greedy equivalence. Small-model results are not
universally faster: final 1B TP=2 warm short decode 246.6 → 239.6 tok/s,
long 219.1 → 213.8, despite retaining its fused native decode. Vendor prefill
and communicator resources are part of this hybrid; the small regression's
precise cause is not isolated.

Logs, the temporary public-API driver, and the detached baseline worktree are
under temp/hybrid-collectives/ and temp/hybrid-baseline/ (not committed).

## Retired decisions (kept so they are not re-litigated)

- **The CPU production path** (host Model/Trunk/components, CPU_AVX2
  kernels, host-io GPU FFN + hipRTC, HardwareClass factories): deleted
  2026-09 after the GPU-resident path passed the full gate suite,
  including fp32-ideal long-case validation. In git history. The
  `cpu_generic` host-block reference impls are retired too — naive
  reference impls for new block types are now done as GPU kernels; the
  host-side math oracle survives as `src/host_math`. Note: the 70B
  (141 GB > one GCD) is intentionally dark until multi-die placement
  lands (§9).
- **CPU-first bring-up** with hand-rolled AVX2 kernels and NUMA tuning:
  served bring-up well (matched OpenBLAS sgemm at 1.2 TFLOP/s); retired
  with CPU-only builds. In git history.
- **No-BLAS verdict**: scoped to CPU BLAS on Zen3; stands there and is
  irrelevant to the device path (vendor BLAS is used where it fits).
- **hipRTC string kernels**: a workaround for avoiding hipcc; retired.
- **Hardware-class fallback chains** ending in a CPU oracle: replaced by
  per-arch selection with loud failure. (The dev-only CPU reference blocks
  that sat alongside are retired — see §3.)
- **FP32 host io boundary**: the fp32 boundary moved to the device;
  precision unchanged, address space changed.
