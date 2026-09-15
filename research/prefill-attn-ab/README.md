# Prefill attention scheme A/B (2026-09)

Controlled experiment: which SIMT architecture should the prefill attention
kernel (S large) use, per head dim? Standalone harness — deliberately NOT
engine code (`build.sh` + hipcc directly; cases regenerable via
`gen_cases.py`, gitignored).

## Schemes

- **v1** — engine baseline: one block per (query, head), 128 threads,
  lane-per-element, block reduction per key row (2 `__syncthreads`/row),
  online softmax.
- **A** — streaming single-pass, thread-per-query, q in registers, acc in
  registers; zero cross-thread traffic. hd=128 form: acc spills to LDS
  (q+acc = 256 regs can't co-reside) — the honest degraded variant.
- **A2** — A with roles swapped (q in LDS, acc in regs) to halve LDS ops.
- **B** — user's two-phase: per 64-key chunk, raw score tile materialized in
  LDS during the dot pass (q staged in regs from an LDS q tile), deferred
  `exp(score − chunk_max)` in the V pass (acc staged in regs from an LDS acc
  tile), one online rescale per chunk. Phase-separated registers; the only
  `__syncthreads` is the q-tile staging. This is the FlashAttention-2
  *structure* (what an MFMA kernel must do).

Case semantics: query s of S attends keys [0, T−S+s]; fp32 throughout;
reference = PyTorch SDPA fp32 with explicit additive mask. Correctness gate
(design agreement): max abs diff < 2e-4 (fp-ordering noise; observed ~1e-7).
**54/54 checks pass for all schemes** (18 cases: hd ∈ {64,128}, T 1→2048
incl. non-full-chunk 63/65/130/1000, S ∈ {T, 512, 65, 1}).

Timing: H=32 heads to fill 110 CUs the way real models do; adaptive iters
targeting 0.3 s/measurement; KV GB/s counts per-query bytes touched
(L1/L2 dedup makes >1.6 TB/s possible).

## Results (ms/call, lower is better; single GCD, gfx90a)

| hd=64 case | v1 | A | A2 | B |
|---|---|---|---|---|
| T=1000 S=1000 | 16.1 | **3.10** | 4.72 | 8.84 |
| T=1000 S=512 | 12.6 | **3.05** | 3.05 | 6.55 |
| T=2048 S=512 | 30.8 | **6.25** | 6.27 | 14.0 |
| T=4000 S=512 | 65.1 | **11.99** | 12.05 | 27.1 |

| hd=128 case | v1 | A | A2 | B |
|---|---|---|---|---|
| T=1000 S=1000 | **16.65** | 17.7 | 37.0 | 22.9 |
| T=1000 S=512 | **12.99** | 14.98 | 29.7 | 17.97 |
| T=2048 S=512 | 31.67 | **31.51** | 63.7 | 38.6 |
| T=4000 S=512 | 66.6 | **61.98** | 128.3 | 85.4 |

## Verdict

- **hd=64: scheme A, decisively** (4–5.4× v1, 2.2–2.9× B). Full-register
  thread-per-query with zero LDS is the right SIMT prefill kernel for the
  1B (and anything hd=64).
- **hd=128: SIMT is tapped out** — v1 ≈ A (~1 TB/s effective). A's degraded
  acc-in-LDS form eats its advantage; A2 shows why LDS forms lose harder
  (see below). Keep v1 for hd=128 prefill **until the MFMA variant**: the
  score-tile-must-materialize shape (B's structure) with acc living in the
  MFMA accumulator register file is the real hd=128 answer, not more SIMT.
- **B loses the SIMT round at both head dims.** Mechanism: LDS appetite
  (q + score + acc tiles) caps occupancy at 1 block/CU (hd=64) or forces
  half-dead wavefronts (hd=128, QT=32 in 64-thread blocks). Its loss here
  says nothing about its MFMA form, where the tile structure is mandatory.

## Lessons

1. **LDS bank geometry decides hd=128**: per-lane rows at 128-float stride
   put all lanes in the same bank (128 ≡ 0 mod 32) → 32-way conflicts on
   every access. A2 (q-in-LDS) is 2× slower than A (acc-in-LDS) at hd=128
   despite half the LDS ops; at hd=64 the same pattern is survivable
   (A2 ≈ A). Any future LDS-resident layout at hd=128 needs padding or
   swizzling.
2. v1's per-row block reduction costs ~2× more at hd=64 than hd=128
   (half its lanes idle) — which is why the A win is concentrated at hd=64.
3. Dead-thread guards belong in the kernel, not the launcher: the first
   version launched 64 threads with QT=32/16, and threads ≥ QT wrote
   per-thread LDS rows out of bounds, corrupting neighboring tiles
   (deterministic, hd=128-catastrophic, hd=64-subtle).

## Next

Engine integration: prefill dispatch to A at hd=64 (v1 retained at hd=128
and for decode via split-T v2). Then the gfx90a MFMA two-phase variant for
hd=128 — B's structure, accumulator-resident acc, possibly fp16/bf16 KV
(user-noted: nothing requires fp32 KV).
