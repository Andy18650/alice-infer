# The Alice Model Format — RC1

Status: **Release candidate 1** (2026-09). Design decided, nothing
implemented. Everything in this format is **compatibility-breaking until
the first loader exists and survives contact with real models**; the
breaking-change window is deliberately held open for that reason. This
document is the normative record of the design; `models/llama-3.2-1b-instruct.jsonc`
is the companion example (its comments are discussion notes, not spec).

Authority note: all design decisions herein are the user's, made in the
2026-09 design sessions; items marked **[open]** are not yet decided;
items marked **[agent rec]** are agent recommendations awaiting the
user's call.

---

## 1. Motivation

Neither safetensors nor GGUF offers one place that answers *"what does
this model look like?"* — today that answer is scattered across a model
card (hope it's filled in), a `config.json` (hope the keys are
recognizable — it is not standardized), or the safetensors header
itself, which requires `xxd` + `head` + `tail` + `jq` gymnastics to
read and says nothing about composition anyway. The alice model format
makes the self-description first-class: a canonical, signed, human-
greppable descriptor that states the model exactly as the engine sees
it, with every byte of every data file accounted for.

The format also fixes, by construction, the specific sins of
safetensors enumerated in the design sessions:

| safetensors defect | alice format answer |
|---|---|
| header trusted, never verified | signed descriptor + per-record BLAKE3 |
| flat name→offset map, no structure | typed members, block specs, composition |
| sharding is an external side channel | physical records `{file, offset, size}` — packing is a non-concept |
| no quantization/codec vocabulary | `codec` per record; type-level storage in VRAM dtype |
| mmap-first design | pread/offset-first design |
| no versioning | version on every member + registry-owned block type versions |
| pickle-adjacent trust problems | bytes are never executed; nothing is derived from untrusted code |

## 2. Principles

1. **The file is the contract.** The descriptor carries composition,
   types, configs, roles, and placement. There is no external manifest
   for files in this format; the engine's manifest machinery is the
   *loader* of this contract, not a second source of truth.
2. **No names.** The old name/role split (safetensors name ↔ engine
   role, bridged by a manifest) is gone. Roles are config keys owned by
   block types. Nothing in the file is a lookup key into a foreign
   vocabulary; nothing in the engine parses meaning out of strings.
3. **Two layers, two jobs.** The descriptor is *meaning* (signed);
   data-file byte ranges are *content* (hashed per record). Corruption
   is caught by record hashes; adversaries are caught by the
   descriptor signature. The reference to every extent lives inside the
   signed region — an attacker must forge a signature to reinterpret
   bytes, not merely recompute hashes.
4. **Types own their configs.** Each block type (with version) defines
   its config schema and is the sole parser of it. The format never
   interprets a type's config fields.
5. **VRAM dtype is the type's contract; disk codec is the loader's
   business.** A block type name (`dense_ffn_bf16`) states what the
   kernels consume. How the bytes are encoded on disk (`codec`) is
   resolved in staging (pread → codec decode → H2D) before any block
   exists. The two registries (block types, codecs) never touch except
   through a codec-legality check keyed on the record's dtype.
6. **No language in the format.** Cross-field relationships are not
   expressible in JSON, deliberately: type-internal derivations
   (`head_dim = io_dim / num_q_heads` where the type-version says so)
   live in layout functions; cross-member invariants (vocab/hidden
   consistency) are *checked by the loader's validator*, not written in
   the file. Signatures freeze the artifact, so any expressiveness
   would be authoring-time only anyway — the shipped file is dumb,
   canonical JSON.
7. **Loud everything.** Unknown type, unknown version, unknown codec,
   hash mismatch, signature invalid, unsigned-by-policy — every failure
   is a hard, named error. No silent fallbacks, ever.

## 3. Vocabulary

- **Type** — `(algorithm, VRAM dtype)`. E.g. `dense_ffn_bf16`,
  `gqa_attention_bf16`, `plain_embedder_bf16`. Registered in the
  engine's block registry. The VRAM dtype is part of the type's
  identity because it is part of the kernel contract.
- **Spec** — a named, bound `{type, version, config}` declaration in
  `block_specs`. Blocks reference specs by name. A spec is followed
  exactly; it is not a template to override.
- **Composition** — the ordered `blocks` list. List order is execution
  order.
- **Member** — a non-block component: chat_template, tokenizer,
  embedder, language_head, trunk. Every member also carries
  `{type, version, config}`.
- **Physical record** — `{file, offset, size, codec, hash}`: where the
  bytes of one member or block live and how to check them.
- **Role** — a config key naming a weight within a type's layout
  (`up`, `gate`, `down`). Owned by the type-version.
- **Layout function** — the contract each `(type, version)` provides:
  `config → [(role, dtype, shape, byte offset)]`, plus total size.
  Config must be *fully consumed*: every field read, every needed field
  present, or the file is invalid.

## 4. The descriptor

Plain, dumb, **canonical JSON (RFC 8785)** — single line, no NULs,
byte-stable across implementations. Top-level fields:

```
name                 display name
version              descriptor format version (this schema's version)
producer             who trained the base model
packager             who produced THIS artifact (conversion, quantization,
                     fine-tune); the signature verifies against this party
                     [open: key fingerprint field name, see §7]
license              SPDX id where one exists, else LicenseRef-*; packager
                     is responsible for the license being legitimate
license_url, license_text   optional, informational
max_context          the model's trained maximum context length (hard fact;
                     bounds KV allocation and user-supplied ctx)
chat_template, tokenizer, embedder, language_head, trunk
                     members, each {type, version, config} + physical record
block_specs          named specs, each {type, version, config}
blocks               the composition: ordered list of {spec} + physical record
```

Norms:

- **Members**: same `{type, version, config}` discipline as block
  specs. Members are versioned like everything else — version things
  that change rarely but fatally.
- **Tied language head**: `tyring`/`tying` is a *member-level* field of
  language_head (not type config — either checking rule needs it to be
  semantically visible). Check is **hash-only equality with the
  embedder's record**: equal hash → pass. Storing the same weight
  twice under different placements is legal; two different codecs
  producing equal-size, hash-equal, usable heads is a hash-collision-
  class event and is not defended against.
- **Restatement is checked, not eliminated**: `vocab_size` appears in
  tokenizer/embedder/language_head, `hidden`-class dims in five places.
  This is deliberate explicitness; the loader's validator must
  cross-check every repetition and hard-fail on any disagreement
  (one-of-many-watches, all watches audited).
- **Cross-member invariants are validator rules, not file syntax**:
  the composition is a typed dataflow graph (each type declares its
  interface dims); the validator binds it. `head_dim` may be omitted
  from a config only where the type-version derives it
  (`io_dim / num_q_heads` for llama3-family GQA v1 — NOT for all GQA).

### Physical record

```
file     path relative to the descriptor
offset   byte offset within that file
size     stored (post-codec) byte count; equals the layout function's
         derived size while codec == "raw" — audited
codec    "raw" in RC1; the only legal value; anything else is a loud error.
         Reserved hook for transport codecs (§6)
hash     BLAKE3 over exactly [offset, offset+size)
```

- `offset`/`size` are unconstrained (no alignment or multiple rules —
  packers may align for speed; the loader's pread path does not care,
  staging memcpys into pinned buffers anyway).
- Blocks may share files freely; multiple blocks per file via offsets
  is the norm. **Single-file models**: descriptor file + `\n\0` + block
  data appended; records simply point past the gate at the same file.
  This is the same format, not a variant.

## 5. Block types and versions

- **Versions are global and registry-owned.** `dense_ffn_bf16 v3`
  means the same layout in every file ever written; the authority is
  the engine's block registry: each entry is `(name, version → layout
  function)`.
- A version pins, as a unit: the config schema, the layout function
  (role set, role order, per-role dtype, packing, alignment), and the
  semantics of config fields. Config schema and byte layout cannot move
  independently.
- Engine policy: support a *set* of versions per type. Unknown version
  = loud load error. No version has shipped, so v1 gets to be clean.
- **Quantized VRAM storage** (`i8 g128` etc.) will be *new types with
  new kernels* — an engine campaign (design §11 direction: weights as
  int8+scales in HBM, in-register dequant before GEMM, like the
  existing GEMV's fp32→bf16), not a format change. Dequant-at-load as a
  compatibility feature is acknowledged worst-of-both-worlds (low
  precision, low performance); if it ever exists it must be loud or
  user-requested — never automatic. Offline dequantization in Python is
  the preferred compat path.

## 6. Codecs (transport encoding)

- RC1 ships exactly one codec: `raw`. The field is reserved on every
  record (same reasoning as `size`).
- Future codecs, if any, are a **few standardized, loader-side
  formats**, applied in staging (`pread → decode → pinned → H2D`),
  dtype-aware (e.g. the sealed rot-split scheme is FP-only), and never
  a block-type concern. Adding a codec touches zero block code.
- Sealed decision (2026-09): lossless bf16 codec = rotate each bf16
  left by 1 bit (`[sign|exp|mant] → [exp][mant|sign]`), split the two
  byte streams, entropy-code the exponent stream (~2.5 bits achieved
  with rANS), store mantissa+sign raw (~10.5 bits/weight total, ~34%
  saving). NOT implemented, NOT to be integrated until reopened.
  Measured context: lossless ceiling on bf16 ≈ 1.5×; blocked formats
  (FP8/FP4 families) deliberately destroy the exponent redundancy this
  harvests, so lossless is marginal there; quantized VRAM storage is
  the real win and is a type problem, not a codec problem.

## 7. Meta-packaging: framing and signing

**File framing.** Every alice-authored file begins with the text magic
`{"ALICE"` (all-caps key) — unique enough, `file(1)`-matchable, greppable,
and it keeps the file valid JSON from byte 0. All-caps is the best we can
do to satisfy `file(1)`: all-caps keys are rare in real JSON (least chance
of a random JSON file being mistaken for a model), and the magic is the
least likely to be disturbed if a tool canonicalizes the whole document
(which is not required but OK). All files (descriptor and data) end their
text region with **`\n\0`** before any binary tail, so: `head -n 1 | jq`
works on any alice file (canonical JSON is one line), and the region
before the first NUL is a valid C string.

- **Descriptor file**: `{"ALICE"`-magic JSON envelope, canonical JSON,
  then `\n\0`, then optionally the model's binary tail (single-file
  models).
- **Data files**: no format requirement at all — the contract is per
  record (`{file, offset, size, hash}` against the signed descriptor);
  any container (or none) is legal. A conventional header
  `{"ALICE":null}\n\0` is recommended so systems can recognize alice
  data files and so block bytes never begin at offset 0 (a block
  payload that is also a valid ELF is a real, if impractical, attack
  surface — and one unrelated to this format's specific design, since
  any format with header space has it; noted, dismissed as
  low-priority). A package uuid in data-file headers binding them to
  their descriptor (catching mixed-version directories) is **[open]**.

**Envelope and signature.** The signature is embedded in the file,
JSON-representable, at a fixed displacement so verification can proceed
without trusting the JSON parser (the paranoid path: locate signature
and signed bytes by displacement, verify, only then parse). Two shapes
were tabled; **[agent rec]** is the second:

1. `{"ALICE": {<model map>}, "signature": "<ed25519>"}` — signature at
   fixed displacement *backwards from the `\n\0`* (Ed25519 is
   fixed-length; a padded hex/base64 encoding is fixed-length too).
2. `{"ALICE": "<signature block>", "model": {<model map>}}\n\0` — with
   the signature block extended to carry `{alg, key fingerprint, sig}`
   as fixed-length fields, *everything* before the variable-length
   model map is fixed-size: signature, key id, and algorithm all sit at
   fixed displacements from the file head. Recommended: verification
   before parse is clean, and "ALICE" reads as the engine's seal.

The signature covers the **canonical (JCS) bytes of the model map** —
nothing else needs covering; per-record hashes already cover content,
and the envelope's remaining fields are exactly the verification
metadata itself.

**Algorithm**: Ed25519 (64-byte signatures, no parameter pitfalls);
BLAKE3 for record hashes. **[open: exact encoding — hex vs base64 —
and field names.**

**Verdict at load** (trinary — this replaces any notion of
"unsigned"):

- `trusted` — signature valid AND key fingerprint pinned in the local
  keyring → load.
- `signed, unknown key` — signature valid, key not pinned → load with
  a loud declaration.
- `invalid` — signature does not verify → hard stop.

Engineering builds sign with the developer's own key, which is in the
developer's own keyring — the code path is uniform, every file carries
a signature, and trust is purely a local-keyring question. Key
rotation: new key, re-sign, republish; the local keyring is
append-only (historical keys kept) since artifacts are immutable.
Revocation machinery (à la TUF) is explicitly out of scope for now.

**Chain of trust is deliberately the back of the priority list.** The
signer proves responsibility, not goodness; the "exfiltrating
fine-tune" threat is only answered by *whose* key you pinned.

## 8. Load-time verification procedure (normative order)

1. Read the fixed prologue; check magic `{"ALICE"` and format version.
2. Locate signature + key fingerprint by displacement; verify Ed25519
   over the canonical model-map bytes. Trinary verdict as above.
3. Parse the (now authenticated) descriptor.
4. Validate: every member/spec type+version known to the registry;
   configs fully consumed; layout functions derive sizes equal to
   declared `size`; codec ∈ {raw}; cross-member invariants (vocab ×3,
   hidden ×5, composition interface binding); tying hash equality.
5. Load records: pread `[offset, offset+size)` → BLAKE3 → decode codec
   → pinned → H2D. Per-record verification during staging; mismatch =
   loud abort.

## 9. Example

See `models/llama-3.2-1b-instruct.jsonc` (draft with discussion
comments; the shipped form is plain canonical JSON).

## 10. Parked / deferred

- Signing infrastructure (keyring format, tooling, distribution of
  packager keys) — after the model loads.
- Expert placement (`expert_freq`): baked list = deterministic initial
  placement (checksummed artifact); runtime remeasurement/reordering =
  engine state in memory, never persisted to the model. Resolves the
  static/dynamic tension; no per-load sidecar structures.
- MoE block types, GLA/linear-attention types, rope-as-block vs
  rope-in-attention-config (composition generality question).
- Codec registry and the sealed rot-split scheme (§6).
- `$schema` field: rejected (hosting URLs is a commitment; the schema
  authority is magic + version + registry).
- The interface/type-checking validator (§4 cross-member invariants)
  is part of the loader work, specified here only as obligations.

## 11. Relationship to the current engine

Nothing in the engine changes for RC1. The existing manifest machinery
remains the loader for safetensors-format models. When the alice
format's loader is built, it consumes the descriptor directly — the
manifest becomes the safetensors-only path (and the experimental
HF-directory synthesis in `Manifest::parse`, commit 68e409f, is the
historical record of why the contract moved into the file).
