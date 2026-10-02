# Alice Inference Engine — Design

Status: living document. This revision (2026-09) commits the engine to
GPU-resident execution and retires the CPU-first bring-up architecture.
The tree is mid-migration; AGENTS.md tracks the state of play.

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
    trunk.read(i)                       // residual → io, apply norm (SIMT)
    block.forward(stream)               // in place on the device stream
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
- The shared stream buffers (residual, io) are device allocations owned
  by the Trunk, `[max_prefill_chunk, d]` fp32. One more cross-block
  sharing instance exists (2026-09, first §14 scratch-pool use): the v4
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
  implementation's private business. Host threading (OpenMP, static
  scheduling only) is confined to the load path and host glue.

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
