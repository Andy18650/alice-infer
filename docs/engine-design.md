# Alice Inference Engine — Design

Status: living draft. Captures decisions to date; the previous code skeleton is
being rebuilt from this document.

## 1. Purpose & target

- From-scratch C++20 LLM inference engine. CPU-first, GPU later.
- Bring-up model: **Llama-3.2-1B-Instruct**
  (`/opt/models/Llama-3.2-1B-Instruct/`, single-shard bf16 safetensors,
  16 layers, hidden 2048, GQA 32 q / 8 kv heads, head_dim 64,
  intermediate 8192, vocab 128256, tied embeddings — lm_head aliases
  embed_tokens, so manifest aliasing is exercised on day one — llama3
  RoPE scaling, three EOS ids). Same LlamaForCausalLM architecture as the
  original target, at bring-up size.
- Scale target (for sizing decisions): DeepSeek-R1-Distill-Llama-70B
  (bf16, 80 layers, hidden 8192, GQA 64/8, head_dim 128,
  intermediate 28672, ~141 GB in 17 shards at /opt/models/deepseek-ai).
- Build: CMake. Minimal core dependencies (nlohmann/json confined to .cpp
  files; GoogleTest for tests; BLAS behind a wrapper).

## 2. Deployment assumptions

Single user. One live context at a time; every new input is appended to *the*
context. No prefix matching between requests. No rollback requests at the
engine level (the session API provides limited truncation via snapshots).

Consequences:

- No multi-context registry or handle indirection. Stateful Blocks own their
  state directly, allocated once, sized for max context length.
- State operations reduce to two verbs: write-during-forward and
  `reset_state()`. No fork/clone/truncate machinery.
- No locking anywhere; nothing is shared between concurrent sequences.
- Memory planning is fully static at load time.

If multi-session ever arrives, it is bolted on by instantiating state per
session — compute never rearchitects, because state lives behind Block
boundaries.

## 3. Source-of-truth philosophy: model = program, manifest = contract

- The engine consumes safetensors directly. GGUF is out of scope (for now).
- An external **manifest** describes model composition: per-Block type and
  weight roles `{name, shape, dtype}` (or `{alias_of}` for tied/aliased
  tensors). Hand-written first; a generator from HF `config.json` later.
- The manifest is the *total contract*: `validate()` performs a two-way audit
  against the actual weight source (missing / shape / dtype / duplicate claim
  / unclaimed tensor / alias resolution) and reports all errors at once.

## 4. Tensor & dtype contract

- **Dtype is static at type level.** `TView<T>`: non-owning, rank ≤ 2
  (`[M, N]` row-major; rank-1 tensors load as `1×N`), element type in the
  type — no `void*`, no runtime dtype field, no `as<T>()` (aliases
  `F32View`, `BF16View`; `bf16` is a strong storage struct with explicit
  fp32 conversion). A wrong-dtype view is a compile error, not a runtime
  check.
- Runtime dtype survives ONLY in `safetensors::TensorInfo` / manifest
  `WeightSpec` (the `DType` enum) — file reality. The single checked
  runtime→static conversion is `File::view_as<T>()` at the load boundary.
- No owning Tensor class, and no polymorphic view hierarchy: quantized
  formats get concrete COMPOSITE types (data + scales structs) named by
  their abstract block class's constructor contract (§11).
- bf16 storage with fp32 accumulation is baked in as the numerics contract.
- Prefill vs decode is **shape dispatch inside kernels** (M==1 → GEMV path),
  not tensor generality. (Whether the interface also *explicitly* separates
  prefill/decode is an open question — see Open Questions.)

## 5. Component taxonomy

- **Functional** — fully stateless primitives, `torch.nn.functional`-style:
  everything needed (weights included) is passed per call. Layers/Blocks may
  bypass them with fused implementations.
- **Weightful** — owns weights; immutable after load; context-independent
  (FFN, projections).
- **Stateful** — owns state not derivable from weights (attention KV cache,
  SSM recurrent state).
- **Block** — smallest unit the manifest describes; self-contained compute
  module (weightful or stateful).
- **Embedder** — token ids → FP32 vectors. Sibling interface (its input is
  ids, not an FP32 chunk).
- **LmHead** — FP32 hidden → FP32 logits, different size in/out; deliberately
  NOT bent into the Block signature for one outlier. At prefill only the
  final hidden row is projected by default.
- **Sampler** — FP32 logits → token id. See §7.3.
- **Trunk** — owns the residual-stream buffers, the static Block-boundary io
  buffer(s), ALL connection weights (norm gammas now; HC mixing /
  attention-residual gather weights later), and the wiring scheme. See §7.2.
- **Model** — container + wiring loop. Owns its parts directly via
  `unique_ptr`, fail-fast checks at construction. Knows no architecture
  specifics. Top-level contract: token ids in → token id out.

## 6. Forward loop

All inter-Block data is FP32, host memory, `[S, d]` row-major, carried in
ONE static io buffer owned by the Trunk. `forward` is in-place: the Block
reads its input from io and writes its output into the same io — an explicit
"no per-layer boundary buffers" contract.

```
io = embedder.embed(tokens)        // ids → FP32 [S, d], into the io buffer
for block in blocks:
    trunk.read(i, io)              // residual → io, apply normalization
    block.forward(io)              // in-place: io in, io out
    trunk.write(i, io)             // residual += io
logits = lm_head.project(trunk.output(io))   // final norm; last row at prefill
token  = sampler.sample(logits)
```

Prefill processes the *entire* prompt including the last token; the logits at
the final position are the distribution for the first generated token.
Invariant: **every token enters Block state exactly once, and its logits are
available immediately after processing.**

Position never crosses any interface. Each stateful Block keeps a private
position counter, incremented per token processed (KV write offset; RoPE
phase = base + arange(S) for a prefill chunk). The counter is NOT the state
length: a Block may evict and compact its state at will (KVPress-style),
leaving its state shorter than the position — strictly the Block's own
business, invisible to everything outside it. RoPE tables are private
buffers of the attention implementation, built at construction from
manifest config.

## 7. Class structure

### 7.1 Blocks

- `Block` — abstract interface: `void forward(TensorView io)`, virtual
  destructor. Boundary contract: FP32 host data, in-place.
- `StatefulBlock : Block` — adds `reset_state()` (needed now: new session),
  plus declared-but-unimplemented snapshot/stash methods with default traps:
  `serialize_state`, `snapshot_create`, `snapshot_rollback` (future revision;
  see session-api.md).
- One **abstract class per (block type × weight dtype)** — e.g.
  `DenseFfnBF16`. Even when the compute graph is identical across dtypes,
  each gets its own abstract class. Rationale (anti-Tensor-class): dtype-in-
  data plus graph-in-code forces per-dtype switches with per-architecture
  ifdefs in every block (e.g. NVFP4 fallback off sm_100+). Fixing dtype at
  the type level keeps each implementation a single switch-free unit.
- One **concrete `final` implementation per hardware class**
  (`CPU_GENERIC`, `CPU_AVX2`, `CPU_AVX512`, `CPU_AMX`, `SM_90`, …).
  `CPU_GENERIC` is the naive reference implementation — a first-class
  citizen and the permanent test oracle, not scaffolding.
- **Constructor contract** (identical signature across all implementations
  of an abstract class; enforced by convention + factory, since constructors
  can't be virtual): takes *already validated* weight views + `EngineConfig`;
  allocates private buffers (weights, transients, state if any); copies/
  converts/packs weights into its own layout (CPU impls may pre-pack for
  their kernels; GPU impls stage H2D). Passed weight views are considered
  freed the moment the constructor returns.
- Hardware selection: one factory per abstract class; runtime detection
  (cpuid / CUDA) with an explicit fallback chain. Kept deliberately shallow —
  CPU-only for the foreseeable bring-up period.
- Construction-time, engine-wide parameters (`max_seq_len`;
  `max_prefill_chunk` — requests beyond → hard error; hardware override)
  are passed to the factory/loader; an `EngineConfig` struct is just
  packaging for these, not a designed object yet.
- Implementations may reuse transient memory *internally* — e.g. chunked
  FFN: tile over the intermediate dim (gate-chunk · up-chunk → SiLU-mul →
  partial down-proj → accumulate running sum), the flash-attention tiling
  idea without online softmax. The running sum lives block-privately (one
  extra memcpy vs. the in-place contract — accepted).

### 7.2 Trunk

- The "where the mess lives" class — the `XXForCausalLM` analog. A model
  proposing a new residual scheme gets a new Trunk implementation, by design
  (the technology argument: same as a new attention or FFN type).
- Owns: residual-stream buffer(s), the static io buffer(s), norm weights,
  norm compute, and the residual wiring.
- **Normalization is merged into the Trunk**, deliberately: norm × residual
  combinations risk blow-up, per-layer Normalizer objects are excessive
  (mainstream models use one norm type uniformly), and callback plumbing is
  rejected. Revisit triggers: trunk-impl proliferation, mixed-norm models
  arriving, CPU-side norm appearing in profiles.
- First implementation: RMSNorm + plain add-residual (Llama family).

### 7.3 Model, Embedder, LmHead, Sampler

- `Model` owns: Embedder, Blocks, Trunk, LmHead, Sampler. Stays wiring-only.
- These components are **thin concrete classes now** — no dtype×hardware
  abstract matrix until a second implementation exists (Lesson 101: do not
  over-architect). Algorithm-level extensibility (n-gram embeddings, exotic
  samplers) is future abstract-class territory.
- `Sampler` is contained in Model (top-level: ids in → id out), with
  sampling config fixed at construction/session start (mirrors the session
  API, where temperature is an auxiliary new_session parameter). Temperature
  0 (= argmax) for tests. Setters may come later for an OpenAI-compatible
  endpoint; expected to be rare (providers converge on fixed recommended
  sampling configs — the policy distribution is shaped by post-training at
  those settings; e.g. reasoning models repetition-loop at temperature 0).
  The Sampler owns the RNG — the first mutable non-weight state in the
  pipeline; acceptable under single-session.

### 7.4 ModelLoader

- Factory: model path + manifest → `Model`. Scope: safetensors only.
- Shape: a narrow entry function that sniffs format by *content* (magic
  bytes / directory shape) and delegates to a per-format loader
  (`SafetensorsLoader` now). Registry machinery deferred until format #3.
  (Standard practice: sniff-then-delegate — LLVM `identify_magic`, image
  libraries, GGUF magic. A big switch inside one function is not.)

## 8. Memory architecture

- **No allocation during `forward`.** All buffers are acquired at
  construction time.
- **Private buffers.** Every Block/Trunk/component allocates its own weight,
  transient, and state buffers in its constructor, in whatever dtype/layout/
  location it wants. Transients are private *for now* — revisited if memory
  pressure demands (documented option: shared scratch with static offset
  reuse across sequential Blocks, ~100 GB → ~2–3 GB at 8K prefill for the
  70B; and per-block internal reuse patterns like chunked FFN, §7.1).
- **Static boundary buffers.** The Trunk owns the residual stream and the
  single static io buffer `[max_prefill_chunk, d]` FP32 — the purely static
  form of scratch sharing, applied exactly where we have control: Block
  boundaries. Ownership caveat: the io buffer must be handed from the Trunk
  to Embedder/LmHead/Blocks; if that plumbing proves awkward, ownership
  moves to `Model` (flagged, not decided).
- **Load flow**: per Block, in order — `pread` its tensors from shard files
  into staging RAM → constructor copies/converts/packs into private buffers →
  staging freed immediately. Peak load RAM ≈ resident weights + one Block's
  staging. Audit = load-time report of component-reported buffer sizes.
- **No mmap residency** (see §9).

## 9. Weight loading & the mmap decision

- Weights are loaded with `pread` into staging, then privately owned by
  components. mmap appears nowhere in the steady state.
- Rationale (aligned with Colibri's findings — github.com/JustVugg/colibri):
  mmap assumes *re-read*; one-time streaming of 141 GB is the opposite.
  Mapped pages count in RSS/cgroup accounting (fat OOM target), reclaim
  evicts workload-blind, cgroup page-cache charging can OOM-kill when reclaim
  lags, demand faults stall the hot path and serialize on `mmap_lock`, and
  the page cache duplicates data the engine never reuses. If the SSD-
  resident regime ever arrives (models ≫ RAM), the answer is Colibri's
  playbook: explicit user-space cache + async pread pool + io_uring +
  measured O_DIRECT — which fits this design as just another hardware-class
  implementation whose "private buffers" are a cache.
- Safetensors specifics: shards discovered by directory scan; duplicate
  tensor names across shards = error; rank ≤ 2 enforced; headers read for
  manifest validation; manifest-driven assembly.

## 10. Kernels & backends

- BLAS wrapper lives in code: self-declared `extern "C"` prototypes
  (OpenBLAS `cblas_sbgemm` vs MKL `cblas_gemm_bf16bf16f32`); CMake only finds
  BLAS and defines `ALICE_BLAS_BACKEND_*`.
- Threading model: the model loop is strictly sequential; intra-op
  parallelism (threads, BLAS) is each implementation's private business.
  OpenMP and OpenBLAS both keep persistent worker pools (created at the
  first parallel region / first call, reused thereafter — no per-call
  spawn), and neither accepts an injected external pool, so an engine-wide
  thread pool is neither needed nor possible; the real knobs are thread
  count, affinity, and spin policy (`OMP_WAIT_POLICY=active`; cf. Colibri's
  hot-thread tuning). The sequential loop avoids nested oversubscription.
  CPU_GENERIC kernels parallelize with OpenMP over OUTPUT ELEMENTS ONLY —
  every reduction stays sequential within a thread, so results are
  bit-identical to the naive oracle and `OMP_NUM_THREADS=1` reproduces the
  sequential baseline with the same binary. Thread-count sweep
  (tools/omp_sweep.sh) on the EPYC box: decode peaks at one thread per
  physical core — 48 threads ≈ 28.5 tok/s, stable; all 96 ≈ 11–18 tok/s
  and unstable (memory-bound GEMV: SMT siblings only contend). Prefill is
  compute-bound and prefers ~80 threads (≈ 75 tok/s). Tests run with
  OMP_NUM_THREADS=48 via the gtest_discover_tests ENVIRONMENT property in
  CMakeLists.txt (the single discoverable place).
- **Numerics-equivalence policy** (user-signed-off 2026-07): bit-exactness
  is scoped to (a) within-impl determinism per fixed schedule (including
  incremental==batched and thread-count independence) and (b) the oracle's
  own OpenMP parallelization. CROSS-impl equivalence is tolerance-based —
  optimized impls may reorder FP sums; the teacher-forced canonical test is
  the equivalence gate. Rationale: disassembly shows the oracle's decode is
  reduction-LATENCY-bound (~29 tok/s predicted from the scalar `addss`
  chain vs 28.5 measured, NOT bandwidth-bound), and IEEE non-associativity
  forbids the compiler from fixing it; breaking the chain requires
  reordering, which is ours to allow, not the compiler's.
- **CPU_AVX2 status** (first custom+optimized operator, dense FFN only;
  factory chain CPU_AVX2→CPU_GENERIC resolves per block type, cpuid-
  guarded): fully private storage and math (no shared toolbox), per-TU
  `-mavx2 -mfma`. Step 2 (structure, scalar inner loops) landed: aligned
  64B storage with parallel first-touch (NUMA-local residency without
  mbind); **S-dispatch on schedule, never on arithmetic order** — S==1
  uses dot-form with contiguous row streaming (a column-block-parallel
  down-pass measured ~3.5x WORSE at 48T: 64B slices at 4KB stride destroy
  DRAM row-buffer efficiency); S>1 uses the fused gate/up/down chunk loop
  with transposed w_down and a private [S,d] accumulator. Both paths
  accumulate per element in ascending-j order → incremental==batched stays
  bit-exact. w_down kept in both layouts (+50% FFN weight storage, flagged
  for memory audit). Measured (1B, Release): decode 26.7→32.6 tok/s
  (+22%) at 48T; prefill 50.3→30.8 (regression from the column-block
  pattern — accepted, blocked layout / step-3 intrinsics are the fix);
  ~+30% structure win at 1T. Step 3 (AVX2+FMA intrinsics into the
  isolated dot/axpy kernels) is next.
- Kernel workspaces (tiled/swizzled layouts, device buffers) are private
  buffers allocated at construction.

## 11. Extensibility targets (designed for, not built)

- GLM-5.3 Flash: mHC trunk scheme; hybrid KDA (fixed recurrent state) + full
  attention (growing cache).
- Kimi K3: attention residuals (trunk history + gather weights).
- MoE Blocks: many small expert weights behind one Block interface.

- Quantized weight formats (MXFP4, NVFP4, FP8_E4M3 with 128×128 FP32 block
  scales, …). A quantized matrix is a FORMAT (a bundle of tensors + compute
  semantics), not a dtype; each piece lands on an existing layer:
  1. **File reality** — `DType` gains storage dtypes (F8_E4M3, U8, …);
     safetensors parser accepts them. (Today it rejects.)
  2. **Contract** — `WeightSpec` gains a composite variant (format tag,
     logical shape, component sub-specs: data + scales [+ global scale]);
     `validate()` learns format-specific shape relations (e.g. scales
     `[M, ⌈N/32⌉]` for MXFP4, `[⌈M/128⌉, ⌈N/128⌉]` for FP8-B128).
     Two-way audit totality is preserved.
  3. **Semantics** — new abstract block classes per format (the block-type ×
     dtype axis simply gains values). Fallback implementations may legally
     dequantize at load (constructor contract: "copies/converts/packs");
     native implementations keep packed data + scales and compute from them.
  Boundary (FP32 io), Trunk, state, and loader flow are all unchanged;
  mixed-format models fall out naturally since format is per-block-type.
  Testing: per-format tolerance policy (scale-after-accumulate vs
  scale-before) instead of bit-exactness; §4's "bf16 baked in" line and §13
  get amended when the first format lands.

Extension points: Trunk polymorphism; Block type registry in the manifest;
opaque per-Block state layouts; snapshot/stash interface on StatefulBlock.

## 12. Non-goals

- Tokenizer / chat template / detokenizer inside `Model` (frontend's job —
  the CLI/server will own one eventually; the engine repo boundary is ids).
- The generation loop and stop conditions (EOS — the bring-up model has
  three EOS ids — max_tokens) live in the frontend; `Model` provides the
  token-level step. Long-term one frontend serves both CLI and API server;
  for the first PoC the CLI simply owns the entire REPL (accepted
  simplification).
- GGUF (for now). Multi-request scheduling. Fork/beam/speculative state ops.
- Python inside the engine core.

## 13. Testing strategy

- GoogleTest. Naive-vs-optimized kernel equivalence as the core property
  (`CPU_GENERIC` is always available as the oracle).
- Synthetic safetensors fixtures (incl. corruption cases). Manifest
  validation against the real 1B source (later the 70B).
- Determinism: same tokens + fresh state + temperature 0 → identical output.

## 14. Open questions & reminders

- **Prefill/decode separation** — whether the interface explicitly separates
  decode from prefill (beyond shape dispatch). CIRCLE BACK when designing
  SSM/linear attention blocks, whose prefill (chunkwise-parallel) and decode
  (recurrent-step) forms differ far more than attention's. Related:
  speculative decoding blurs the line further — decode becomes 'short
  prefill with frequent rollbacks' needing multiple logits per step; likely
  a new interface when the time comes.
- **Transient sharing** — private now; revisit if memory pressure demands
  (§8).
- **Trunk/norm merge** — revisit triggers in §7.2.
- **FP32 CPU boundary** — review if benchmarks show boundary costs (mHC/
  AttnRes math, H2D/D2H latency) are significant.
- **Model/Embedder/LmHead naming** — LmHead name provisional
  ("de-embedder" candidate); `lm_head` matches weight files.
