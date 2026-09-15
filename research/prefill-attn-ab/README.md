# Prefill attention scheme A/B (2026-09)

Controlled experiment: which SIMT architecture should the prefill attention
kernel (S large) use, per head dim? Standalone harness — deliberately NOT
engine code (`build.sh` + hipcc directly; cases regenerable via
`gen_cases.py`, gitignored).

## Schemes

- **v1** — engine baseline: one block per (query, head), 128 threads,
  lane-per-element, block reduction per key row (2 `__syncthreads`/row),
  online softmax.
- **v3A** — streaming single-pass, thread-per-query, q in registers, acc in
  registers; zero cross-thread traffic. hd=128 form: acc spills to LDS
  (q+acc = 256 regs can't co-reside) — the honest degraded variant.
- **v3A2** — v3A with roles swapped (q in LDS, acc in regs) to halve LDS ops.
- **v3B** — two-phase: per 64-key chunk, raw score tile materialized in
  LDS during the dot pass (q staged in regs from an LDS q tile), deferred
  `exp(score − chunk_max)` in the V pass (acc staged in regs from an LDS acc
  tile), one online rescale per chunk. Phase-separated registers; the only
  `__syncthreads` is the q-tile staging. This is the FlashAttention-2
  *structure* (what an MFMA kernel must do).
- **v4** — split-T prefill (FA-2-style split-KV, user's design): each block
  owns ONE key chunk for a 64-query tile (thread-per-query): raw scores to a
  16 KB LDS tile in key-major layout (bank-conflict-free in both phases, no
  padding), deferred exp with the single-chunk max, partial (acc, max,
  sum-exp) to GLOBAL memory, merged by a small combine kernel. No staged
  q/acc tiles, no running softmax, no `__syncthreads`, fully independent
  blocks -> 4 blocks/CU.

Case semantics: query s of S attends keys [0, T−S+s]; fp32 throughout;
reference = PyTorch SDPA fp32 with explicit additive mask. Correctness gate
(design agreement): max abs diff < 2e-4 (fp-ordering noise; observed ~1e-7).
**54/54 checks pass for all schemes** (18 cases: hd ∈ {64,128}, T 1→2048
incl. non-full-chunk 63/65/130/1000, S ∈ {T, 512, 65, 1}).

Timing: H=32 heads to fill 110 CUs the way real models do; adaptive iters
targeting 0.3 s/measurement; KV GB/s counts per-query bytes touched
(L1/L2 dedup makes >1.6 TB/s possible).

## Results (ms/call, lower is better; single GCD, gfx90a)

| hd=64 case | v1 | v3A | v3A2 | v3B | v4 |
|---|---|---|---|---|---|
| T=1000 S=1000 | 16.1 | 3.10 | 4.72 | 8.84 | **1.59** |
| T=1000 S=512 | 12.6 | 3.05 | 3.05 | 6.55 | **1.08** |
| T=2048 S=512 | 30.8 | 6.25 | 6.27 | 14.0 | **2.34** |
| T=4000 S=512 | 65.1 | 11.99 | 12.05 | 27.1 | **4.68** |

| hd=128 case | v1 | v3A | v3A2 | v3B | v4 |
|---|---|---|---|---|---|
| T=1000 S=1000 | 16.65 | 17.7 | 37.0 | 22.9 | **5.13** |
| T=1000 S=512 | 12.99 | 14.98 | 29.7 | 17.97 | **3.62** |
| T=2048 S=512 | 31.67 | 31.51 | 63.7 | 38.6 | **8.10** |
| T=4000 S=512 | 66.6 | 61.98 | 128.3 | 85.4 | **14.63** |

## Decode-shape coda (S=1/S=4): can v1 be retired?

v2 = the engine's flash-decode kernel ported into this harness (split-T,
in-warp shuffle reduction, fp32 out). Measured at decode shapes (H=32):

| µs, S=1 | T=1 | T=8 | T=64 | T=128 | T=256 | T=512 | T=1000 | S=4 T=512 |
|---|---|---|---|---|---|---|---|---|
| hd=64 v1 | **4** | 12 | 77 | 152 | 303 | 701 | 1526 | 719 |
| hd=64 v2 | 7 | **8** | **22** | **23** | **27** | **36** | **49** | **67** |
| hd=64 v4 | 7 | 13 | 61 | 61 | 65 | 95 | 198 | 99 |
| hd=128 v1 | **3** | 12 | 78 | 155 | 359 | 806 | 1550 | 804 |
| hd=128 v2 | 7 | **8** | **23** | **24** | **32** | **40** | **53** | **76** |
| hd=128 v4 | 12 | 33 | 206 | 212 | 327 | 343 | 669 | 345 |

**v2 wins every decode shape except T=1** (1.5× at T=8 → ~30× at T=1000);
v4 at S=1 is structurally degenerate (1 live lane of 64) and loses as
predicted. T=1 favors v1 by ~3 µs of pure launch overhead (v2 runs two
kernels), once per session — not worth a dispatch branch. **Engine
dispatch therefore becomes two clauses, no v1: S>4 -> v4, S<=4 -> v2 at
ALL T** — the T>=256 threshold was as unvalidated as the S<=4 one.

## Verdict

**v4 (split-T prefill) wins everywhere, decisively.** 10–14× over v1 at
hd=64, 3.6–4.6× at hd=128; 2.6–2.8× over v3A at hd=64, 3.5–4.2× at hd=128.
Effective KV throughput 5.2–6.7 TB/s (hd=64) / 3.2–4.3 TB/s (hd=128) — far
above HBM peak, i.e. the duplicated K/V streams across chunk-blocks are
served by L2 as predicted (per-head K/V ≤ 4 MB at T=4000 < 8 MB L2).

Why it beats v3A even at hd=64 (where v3A has zero LDS and zero combine):
v4 combines v3B's arithmetic economy (1 exp/key, no per-key acc rescale —
the savings that never paid at v3B's occupancy) with 4-blocks/CU occupancy
(16 KB LDS) and fully independent blocks (no per-query serial chunk chain).
The combine/partials cost is ~3% of K/V traffic — peanuts, as predicted.

Prior round (pre-v4) conclusions, superseded: v3A had won hd=64 (4–5× v1);
hd=128 SIMT had looked tapped out (v1 ≈ v3A) pending MFMA. v4 dissolved the
hd=128 wall by taking the acc tile out of LDS entirely — no MFMA needed for
the next big hd=128 win, though the MFMA variant remains the sequel (v4's
score-tile shape is what GEMM phases need).

## Lessons

1. **LDS bank geometry decides hd=128 layouts**: per-lane rows at 128-float
   stride put all lanes in the same bank (128 ≡ 0 mod 32) → 32-way
   conflicts. v4's key-major score tile (`scores[QT*key + thread]`) dodges
   this for free in both phases — lane-uniform key iteration means lanes
   always write/read consecutive addresses. No padding needed.
2. **Occupancy beats arithmetic accounting** (v3B): 1 block/CU wastes the
   best instruction schedule. **But arithmetic economy pays once occupancy
   is healthy** (v4 vs v3A): 1 exp/key + no per-key rescale is worth
   ~2.5× when blocks actually fill the machine.
3. v1's per-row block reduction costs ~2× more at hd=64 than hd=128
   (half its lanes idle).
4. Dead-thread/edge guards belong in the kernel, not the launcher (the
   v3-round QT/threads mismatch caused deterministic LDS corruption).
5. The S≤4-vs-prefill dispatch boundary in the engine was a reasoned guess
   (never measured); this experiment is the first actual prefill research
   and replaces it: prefill is throughput-bound, and split-T wins there not
   by cutting latency but by fixing per-block arithmetic economy + occupancy.

## Next

Engine integration: v4 for prefill (both head dims), v2 split-T for decode
(S≤4), v1 retained only for short-context decode (T<256). Then the gfx90a
MFMA variant of v4's structure, possibly fp16/bf16 KV (user-noted: nothing
requires fp32 KV).
