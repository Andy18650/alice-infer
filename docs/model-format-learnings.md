# Model Format Round — Learnings

Companion to `model-format.md` (the spec) and `model-loader.md` (the
loader design). This file records what the 2026-09 sessions TAUGHT —
process corrections, design-philosophy calibrations, and operational
traps — so future agents don't re-earn them. The spec says what the
format IS; this says what its development taught about HOW to work on
it. Lessons are ordered roughly by how expensive they were.

---

## 1. The dependency question comes before the build (the BLAKE3 affair)

The most expensive mistake of the round: the format needed BLAKE3, the
box appeared not to have it, and the agent ported the portable
reference implementation in-tree (~450 lines) rather than stopping to
ask. It then debugged a finalize counter bug, discovered the scalar
port ran at 56 MB/s, and wrote a parallel chunk-tree patch to fix
performance it had itself created — the majority of a round's token
budget. `libblake3-dev` was in the repos the whole time; one `apt
install` plus a ~30-line adapter replaced everything, at 4.1 GB/s.

**The rule was already written** (AGENTS.md: when the right solution
needs something outside the sandbox — a package install — STOP and ask;
standard approaches preferred over hermetic workarounds). The failure
was not checking `apt-cache` before concluding the dependency was
absent, and not treating "implement a crypto primitive myself" as a
decision that needed the user.

Corollary, now policy: compile time is not a concern at this scale;
the engine is library-like in the GGML/libcurl sense (consumers build
on it and recompile); so expose and use industry-standard libraries
freely — including in public headers — and delete wrapper layers that
add no vocabulary (the blake3 wrapper eventually thinned to one
`blake3_hex()` helper in the loader and was deleted).

## 2. Constraints have provenance — check whose they are

Mid-session the agent argued against the user's own proposal by
invoking "the house rule: nlohmann private in .cpp only." The rule was
in AGENTS.md — and was NOT the user's. AGENTS.md's own constraints
section says nothing in it is law unless the user said it; an earlier
agent's decision had hardened into enforced policy. The user replaced
it with "use industry-standard dependencies when appropriate; CMake is
the source of truth for links."

**Before citing any rule against a design proposal, know whether the
user enacted it. If provenance is unclear, flag it and ask — the
working agreement requires exactly that.**

## 3. The overengineering pattern: runtime enforcement of what review owns

Three structures were built and then removed, all the same shape —
runtime checkers for things code review and process already own:

- **BlockConfig read-tracking** (config "fully consumed" enforcement).
  Factories are authored by us or via reviewed PRs; the keys a type
  reads are known at review time. Type/format drift is a GitHub issue,
  not a runtime checker's business. Replaced by per-type structs +
  `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE`.
- **expect_keys** (strict unknown-field rejection). External review
  supplied the decisive argument: strict rejection is NOT fine-grained
  versioning — a field addition locks out old readers exactly like a
  version bump, just implicitly, without changelog semantics, and with
  no way to distinguish intentional additions from mistakes. Its entire
  marginal value was catching a forgotten version bump: a process
  failure. Schema evolution is version-governed; unknown fields are
  ignored; forgotten bumps are the PM's problem.
- **The `where`-string plumbing** (get_u/get_s context parameters,
  parse functions threading labels). Replaced by nlohmann's
  `JSON_DIAGNOSTICS` — exception messages carry the JSON pointer path,
  which is strictly better information than hand-maintained labels.

**Diagnostic reflex to watch for: if a check exists to catch a mistake
that code review or CI would catch anyway, it does not belong at
runtime.**

## 4. "Loud failure" recalibrated

Loud failure means *fail loudly with context the library's own errors
lack* — semantic checks (magic, version, unknown type in registry,
codec, hash mismatch, short read). It does NOT mean check-then-fail
reflexes re-deriving what a library already reports. After the
recalibration, the loader's hand-written checks are exactly the ones
encoding decisions; everything mechanical is a library call whose
exception carries a path.

Related technical facts (nlohmann):
- `JSON_DIAGNOSTICS` must be defined identically in every TU (PUBLIC
  CMake definition; out-of-tree harnesses add it manually) or ODR
  bites.
- Diagnostics paths come from parent links INSIDE the parsed document;
  a copied subtree (e.g. `BlockSpecDecl::config`) is its own root and
  loses the path — that site re-adds context by hand. The members
  round inherits this note for `MemberDecl::config`.
- `json::exception` is not a `std::runtime_error`; if an API documents
  runtime_error, convert at the boundary (one catch at `load_blocks`).
- `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` composes across raw-json members
  and nested macro-defined types — it replaced parse_record,
  parse_member, parse_spec, and the spread code; the standalone
  `Record` struct disappeared entirely.

## 5. Coincidence is not contract

Two checks survived review rounds and were then found to be enforcing
facts that are true only under the current configuration:

- **derived == stored size**: true only because today's codec is raw.
  Under any real codec they legitimately differ. The record-level check
  was removed; the contract is the factory-side reconciliation (decoded
  == derived, exactly), which for raw covers the same mismatch.
- **scratch sizing in the spec**: the v4-prefill-partials formula
  (`nh × mpc × ceil(max_seq/64) × (hd+2) × 4`) is a property of the
  KERNEL IMPLEMENTATION, not of (type, version, config) — the same
  .model must load on engines with different prefill kernels. The
  formula sits in the spec's `scratch_bytes()` today only until a
  proper factory exists (user decision: fix with the factory round).
  The `hd+2` itself: hd output floats + online-softmax (m, l) state per
  partial.

Also caught before landing: folding the io buffer into the
init-time allocator would have welded runtime execution state
(S-dependent, forward-passed) into load-time capacity. Scratch is
capacity; the stream is execution.

**When a check or invariant feels natural, ask what makes it true: the
format's contract, or today's configuration.**

## 6. Verification methodology: compare at the constructor seam

The chicken-and-egg (no test model without a packer, no verified
format without a loader) was broken by writing the Python packer plus
a smoke harness BEFORE handing over the loader. The harness's trick:
both loading paths (the since-retired manifest+safetensors path and the
new format+factories path) converged on the SAME block constructor
signature (the deliberate seam of the design), so running `forward()` on
both and `memcmp`-ing
outputs verifies exactly what can differ upstream — bytes and config
delivery — with zero adapter code. It is output-equivalence
(probabilistic evidence of byte equality), not a direct byte compare;
determinism policy (same impl, same S, same schedule) makes it
bit-exact. 32/32 blocks bit-exact, repeatedly, across all subsequent
refactors — this is what made aggressive deletions safe.

BLAKE3 was verified the same way: cross-checked against the official
implementation across every chunk-boundary size (0, 1, 63/64/65,
1023/1024/1025, ... 1 MiB) plus incremental-vs-one-shot.

## 7. Operational traps (each bit at least once)

- **The stale-binary trap**: the research harness statically links
  `libalice.a` and must be RELINKED after any library change; it
  produced a ghost error from deleted code twice before being caught.
  AGENTS.md already warns about build trees not tracking each other.
- **/tmp is wiped by power cycles** — ate the packed smoke model
  repeatedly. Use the repo-local gitignored `temp/` (now the home of
  `temp/packed1b`, the standing 1B smoke pack; re-packable via
  `tools/pack_model.py`).
- **The venv has no pip; `uv pip install <pkg>` is the sanctioned
  mechanism** (used for zstandard, lz4, blake3).
- Descriptor budget, measured: the 1B descriptor is ~7 KB; the 70B with
  MoE projects to ~32 KB (experts are weights, not config). The 1 MiB
  read cap is a wrong-file guard, not a designed limit. **Estimate
  before inventing magic numbers** — an unexamined 64 MiB constant
  survived two review rounds.

## 8. The archaeology rule

The code documents what IS; git history documents how it got here.
Comments narrating removed code ("REMOVED (2026-09)", "replaces the
earlier X") are noise — future agents can read history on demand. The
exception is negative-space guards: comments explaining why a
seemingly-obvious thing is deliberately ABSENT, when it is at risk of
being re-added wrongly (the no-derived==size-check note; pread-not-
ifstream). Kept only those.

Commit messages carry the full rationale for every removal — the
learnings' first home. This file is the index.

## 9. Inherited by future rounds (do not re-litigate, do not forget)

- **Scratch sizing moves to the implementation** when the factory
  round replaces `ResolvedSpec::construct()` (see §5).
- **The TensorView pin**: the user once concluded `Tensor` alone is a
  sufficient constructor contract; the current `TensorView` was an
  inference from the loader-design discussion, now pinned open pending
  their decision. Do not treat either as settled.
- **The packer is provisional.** Its member-layout decisions (trunk =
  pre-norms in execution order then final norm; tied lm_head = the
  embedder's record verbatim; tokenizer = raw tokenizer.json bytes)
  are format decisions wearing tooling clothes — they must be promoted
  into type layouts when members are designed. A Jsonnet-assisted
  authoring layer is contemplated; requires discussion first. The
  packer, factories, and (implicitly) the harness are three
  implementations of the layout orders — drift is caught only by the
  smoke test.
- **Members round**: `MemberDecl::config` is a copy (severed
  diagnostics path, §4); members have never been interpreted — no code
  has read those bytes back.
- **Signing**: deferred behind "bring the model up first." The trinary
  verdict (trusted / signed-unknown-key / invalid), always-signed
  files, and the local-keyring trust model are decided; Ed25519 and
  the keyring format are not yet chosen.
