# The Alice Model Loader — design (v0, with the format at RC1)

Status: design settled 2026-09 (sessions captured in `docs/model-format.md`,
this file records the loader/allocator layer). First implementation round
follows; the format's breaking-change window stays open until the loader
survives real models.

---

## 1. The load pipeline

```
.model file ──► framing + envelope + magic (read_envelope)
            ──► descriptor parse (canonical JSON)
            ──► SPEC PRE-RESOLUTION    (everything spec-level, before any data file)
            │     • every spec's (type, version) in the registry; config parsed
            │       via the type's struct (missing/wrong keys throw, wrapped
            │       with spec context)
            │     • composition check: all specs present the same interface
            │     • scratch: max over specs' requirements, allocated once,
            │       poison-filled
            ──► BLOCK LOOP — single pass over the composition; per block:
                  check (spec ref, codec == raw, record size == derived)
                  → pread → BLAKE3 → construct
                  (a bad record at block i fails after blocks < i load —
                  accepted: machine-packed files, RAII teardown)
```

Non-block members (embedder/trunk/language_head/tokenizer) are **set
aside** for this round: parsed structurally, not interpreted.

Pre-read validation is scoped to what is derivable *without* the data
files. Post-decode checks (hash, decoded length) require reading; a
corruption failure 10 GB into staging is an accepted cost — corrupted
data is the rare event, descriptor typos are not (models are machine-
packed), and the hash is the guarantee against the former.

## 2. The three ownership layers

Per-block, the bytes pass through exactly one seam with a move, and one
seam with borrows:

- **Loader owns bytes**: preads the record extent into a THP-backed
  (mmap + `MADV_HUGEPAGE`, design §9) chunk, hashes it, decodes it
  (identity for `raw`), and hands the chunk to the factory **moved**.
- **Factory owns layout**: the per-type factory is the embodiment of
  the format's layout function (`config → [(role, dtype, shape, offset)]`).
  It slices the chunk into **borrowed** tensor views and frees the chunk
  when the constructor returns. One chunk per block; no per-tensor
  staging, no streaming — per-block granularity is the validated unit.
- **Constructor owns math**: the block constructor receives borrowed
  tensor views (ptr + dims, the house vocabulary for borrowed host
  access), performs its own H2D into private device storage, and never
  frees host memory. This is why device-block constructors take
  `TensorView<T>` rather than owning `Tensor<T>`: they only *read*
  weights during construction. (The host-side math oracle in `src/host_math`
  keeps owning `Tensor<T>` when it needs host data for compute — e.g. the
  RoPE tables. Owning is correct there; borrowing is correct here.)

The uniform factory signature enables the registration pattern when the
type zoo grows:

```
factory(chunk: bytes──moved──►, spec: ResolvedSpec, allocator,
        context, engine_config) → unique_ptr<gpu::Block>
```

All factories share it; all constructors of implementations of one type
share that type's tensor-typed signature. Implementation selection
(env/CLI knobs, per-arch impls, the CPU-reference escape hatch) is the
factory's private business, later.

## 3. Config: raw json at the handoff, per-type structs at the parse

Specs are pre-resolved before the block loop: name → parsed config +
derived facts (layout bytes, scratch bytes), computed once, shared by
all blocks referencing the spec.

The config crosses the loader→type handoff as raw `nlohmann::json`
(dependency policy, 2026-09: industry-standard libraries when
appropriate). Each type's resolve function declares a config struct
(`NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE`) and parses in one
`config.get<Struct>()` call — missing keys and wrong types throw at
load, wrapped with spec context by the loader. Unknown extra keys are
ignored: if the packager (human or script) inserts an entry the type
doesn't read, that is harmless by policy; if the type's key set and
the format drift apart, that is a code-review / GitHub-issue concern,
not a runtime checker's (factories are authored by us or reviewed PRs —
the keys read are known at review time).

There is deliberately no read-tracking or consumption check. The
earlier `BlockConfig` class (flat map, typed accessors, read-set
diffing) was removed as overengineering (user decision, 2026-09).

Config values are scalars in practice (the descriptor keeps rope
parameters flattened), but nothing forbids nesting — a future type
that wants a sub-object simply declares one in its struct.

## 4. Teardown: RAII or nothing

No bespoke teardown code. Blocks live in `std::vector<std::unique_ptr<
Block>>`; staging is RAII; therefore a throw anywhere in the load loop
(hash mismatch at block 17 included) unwinds the vector and destroys
each block in reverse — every block destructor releasing its private
device memory is a hard obligation of the self-contained-impls
invariant. Known non-C++ hazards deferred with this decision: HIP's
sticky error state after a failed H2D (matters only for retry-in-
process, i.e. the library era).

## 5. The allocator

A per-context object (one per device/thread context; multi-GPU later)
passed down through the factories. Two facilities:

**Named buffers** — `named(name, bytes) → device ptr`: fixed-size state
for layers to communicate beyond the residual stream (user clarification,
2026-10), not temporary workspace. Same name + same size → same stable
pointer. Same name + different size → loud load failure; there is no grow
API. **The name is the sharing contract**: a collision is always intended
aliasing, never an accident to disambiguate. The named-buffer table owns
these allocations until teardown; they are not also owned by the arena /
on-demand allocation ledger.

Temporary device staging for column-major weight repacking is allocated
with `hipMalloc` in each ROCm block's weight-loading constructor and freed
with scoped `hipFree` after the repack completes, also on exception. It
never enters the named-buffer table or survives into inference.

**The scratch buffer** — one anonymous per-GPU buffer, assumed fully
overwritten during each block's `forward()`. Sizing is **max over
per-block scratch requirements**, and the requirement is a **pure
function of (type, version, config, load knobs)** — a registry query
beside the layout function, evaluated during spec pre-resolution. The
allocator then `hipMalloc`s once, **before** the construction loop, and
the pointer is passed as a constructor argument (borrowed, non-owned
member — the same plumbing the v4 prefill partials use today). No
pointer patching, no accept-callbacks: the pointer is valid at the
moment of binding, so there is no temporal window to defend.

The fully-overwritten assumption is testable: the allocator poison-
fills the scratch once at init (NaN pattern), catching read-before-
first-write in the test suite for free.

**Weights are NOT allocated through the allocator** in v1 — blocks
`hipMalloc` their own (immediate allocation for upload). The allocator
already flows into factories, so routing weights through it later
(accounting, OOM context, multi-die placement) needs no signature
change.

**The io buffer is NOT folded into this scheme.** Scratch is capacity
(load-time, fixed); the io stream is execution state (runtime-sized,
flows through `forward(io, S)`). Folding would save nothing — every
block would alias one pointer — while hiding the engine's central
dataflow invariant ("blocks operate on the stream in place") inside
private state. The stream stays Model/trunk-owned and forward-passed.

## 6. Verification at the seam

- **BLAKE3 over stored bytes** `[offset, offset+size)` — verified per
  block during staging, before decode.
- **Exact length reconciliation**: decoded length (concluded by the
  codec from its own data) must **equal** the layout-derived length
  (from the registry) — not greater, not less. Neither is canonical;
  both are independently derived and their agreement is the evidence
  (two-watches: the factory's derived layout vs the descriptor's
  content-addressed record). For `raw`,
  decoded = stored, so this also catches a lying record size. There is
  deliberately NO derived-vs-declared-size check at the record level:
  stored size equals derived size only as a raw-codec coincidence —
  under a real codec they legitimately differ, so treating their
  equality as contract would reject every compressed record
  (user decision, 2026-09).
- **Signature**: framing/envelope handled per `model-format.md` §7.
  Ed25519 verification is deferred with the chain of trust (bring the
  model up first); the loader this round declares signature state
  loudly (present/absent, unverified) and proceeds.

## 7. Deferred (recorded, not forgotten)

- Member loading (embedder/trunk/language_head/tokenizer) and the
  model-level assembly (a new `Model::load` variant for `.model`
  files).
- Implementation selection knobs in factories (incl. any loud/explicit
  storage-cast fallback).
- Multi-GPU: per-thread `hipSetDevice`, device-aware factories (which
  shard of which tensor), placement-aware allocator.
- Prefetch pipeline (read block *i+1* during block *i*'s H2D) — the
  record structure makes double-buffering natural.
- Footprint pre-pass (registry query for device bytes; multi-die
  placement input).
- Ed25519 + keyring; canonical-JSON (JCS) enforcement on the signed
  bytes; signature encodings (hex vs base64 — loader currently parses
  hex hash fields).

## 8. This implementation round (scope)

- BLAKE3 (portable, in-tree, verified against reference vectors).
- Framing + envelope + descriptor parse (blocks + structural members).
- Spec pre-resolution with full-consumption config checks.
- Allocator (named buffers + registry-sized scratch + poison fill).
- Factories + registry entries for `dense_ffn_bf16` v1 and
  `gqa_attention_bf16` v1; the two gpu_rocm constructors take
  borrowed `TensorView<T>` (built from the fetched, BLAKE3-verified blob).
- A Python packer (`tools/pack_model.py`) to break the chicken-and-egg
  (no packer, no test model). The cross-check (packed bytes == HuggingFace
  bytes, every block constructs + forwards bit-exactly) was retired once
  the format path was verified equivalent — recorded in git history.
- Not this round: members, model assembly, signing, tests in ctest
  (manual review first, per the standing decision).
