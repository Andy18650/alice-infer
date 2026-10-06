# Decode roofline study: why the 8B decodes at 34 tok/s, not 100 (2026-09)

Question (user): the 8B decodes at 34.2 tok/s (29.28 ms/token). Weights
traffic is 15.01 GB/token (13.98 GiB: 32 layers × 416 MB + 1.05 GB
lm_head), so at the nominal 1.6 TB/s HBM per GCD the roofline is ~106
tok/s. We are at a third of that. Hypotheses: (a) kernel-launch /
CPU-side bound, (b) insufficient GEMV. Investigate without drastic code
changes; clean explanations over complete solutions.

Method: `bench.cpp` links the engine's **real** kernel objects
(`build*/alice_gpu_kernels.o`, `gpu_rocm_kernels.o`) plus rocBLAS and
replicates the exact S=1 decode kernel sequence (embed gather → per layer
[rmsnorm, convert, QKV gemm, rope, attention ×2, o gemm, add, rmsnorm,
convert, gup gemm, silu_mul, down gemm, add] ×32 → final rmsnorm, convert,
lm gemm, argmax, 4 B D2H). `gemv_ref.hip` adds a pure-read bandwidth
ceiling and a deliberately simple split-K bf16 GEMV (warp-per-row, uint64
loads, 2048-wide K chunks) as the "what a proper GEMV delivers" reference.

Numbers (Release objects, idle GCD 0, MI250X, rocBLAS 5.1.0, 104 CUs):

```
[1] HBM streaming-read ceiling (1 GiB):          1290 GB/s   (81% of nominal)
    -> realistic weights roofline: 15.01 GB / 1.29 TB/s = 11.6 ms = ~86 tok/s

[2] S=1 GEMV per shape (rocBLAS gemm_ex, the engine's exact TN call):
    qkv  [ 6144 x  4096]   0.044 ms  1143 GB/s
    o    [ 4096 x  4096]   0.034 ms   977 GB/s
    gup  [28672 x  4096]   0.323 ms   727 GB/s   <- 32x/token = 10.3 ms
    down [ 4096 x 14336]   0.375 ms   313 GB/s   <- 32x/token = 12.0 ms
    lm   [128256 x  4096]  1.135 ms   925 GB/s
    GEMM-only floor: 26.0 ms/token = 38.5 tok/s

    Same shapes, hand-rolled split-K GEMV: qkv 1030, o 946, gup 1265,
    down 1172, lm 1281 GB/s -> floor 12.7 ms/token = 78.9 tok/s.

[3] SIMT-only chain (324 launches, engine kernels): 2.50 ms/token
[4] Full replica:
    rocBLAS gemms, per-token sync:  29.58 ms/token (33.8 tok/s)  [engine: 29.28]
    rocBLAS gemms, async:           29.63 ms/token  (sync costs ~nothing)
    split-K ref gemms, per-tok syn: 15.48 ms/token (64.6 tok/s)
    CPU enqueue, one token:         1.72 ms/token (452 launches, ~3.8 us each)
```

## Verdict

**The decode gap is the S=1 GEMV path — specifically rocBLAS `gemm_ex`
on the two FFN shapes — not launch overhead.** Both user hypotheses in
their strong form are wrong, hypothesis (b) in substance right:

1. **Not launch/CPU bound.** Issuing a whole token costs the CPU 1.72 ms
   (452 launches) and the per-token sampler D2H adds ~0.05 ms; the full
   SIMT kernel chain is 2.5 ms. Even a zero-overhead host would leave us
   at ~27 ms/token.
2. **The GEMMs are 88% of the token.** 26.0 of 29.6 ms is rocBLAS S=1
   GEMM time, and 22.3 ms of that is just gup (727 GB/s) + down
   (313 GB/s). The attention and lm_head GEMVs are fine-ish (925–1143
   GB/s). The `down` shape ([4096 × 14336], only ~32 N-tiles of work on
   104 CUs, no split-K in the selected kernel) runs at **24% of the
   achievable read bandwidth**.
3. **Headroom demonstrated, not just argued:** swapping ONLY the GEMV
   calls for the naive split-K reference in the identical kernel sequence
   gives **15.5 ms/token = 64.6 tok/s** (1.9× the engine) with CPU
   enqueue still at 1.5 ms. The reference itself is not at the read
   ceiling on every shape (o 946, down 1172), so a tuned GEMV (the
   planned decode-overhead campaign: GEMV + launch diet) plausibly lands
   70–80 tok/s against the ~86 tok/s realistic roofline.

Also measured: the "wide-M1" gemm_ex form (M=1 wide-N) helps `down`
(313 → 606 GB/s) but is no fix; the nominal 1.6 TB/s should be replaced
by the measured 1.29 TB/s streaming ceiling (~86 tok/s) in any roofline
math.

## Files

- `bench.cpp` — replica + decomposition (phases [1]–[6])
- `gemv_ref.hip` — read-BW ceiling + split-K reference GEMV
- `build.sh` — links engine kernel objects from `build-release/`

Rebuild: `./build.sh && ./bench_decode [device]`

## Outcome (2026-09, same study, engine landed)

The study's verdict drove the gpu_rocm S=1 GEMV: a family-private
split-K GEMV in `src/blocks/gpu_rocm/kernels.hip` (warp-per-row, uint64
bf16 loads, partials + combine, in-register operand quantization so the
separate f32_to_bf16 launch disappears), dispatched at S==1 in both
blocks; gemm_ex stays for S>1. CTA target swept via this bench's [6]
section: 1024/2048/4096/8192 -> 13.37/13.37/12.99/14.44 ms/token floor
(4096 kept: qkv 1047, o 933, gup 1258, down 1074, lm 1269 GB/s).

Engine results (Release, single GCD, greedy, default host placement,
all 62 tests green incl. canonical gates; generated text token-identical
to the pre-change baseline on the 8B probe):

| model | decode before | decode after | context |
|---|---|---|---|
| 8B | 34.2 tok/s | **64.9 tok/s** | ~135 |
| 8B | 28.2 tok/s | **47.1 tok/s** | 3770 |
| 1B | 114.6 tok/s | **261.7 tok/s** | ~135 |
| 1B | 86.1 tok/s | **159.6 tok/s** | 3770 |

Prefill unchanged (same code path for S>1): 8B 2182 tok/s at 3640 ctx,
1B 8715. Remaining decode budget: lm_head still on gemm_ex (~1.1
ms/token), SIMT tail ~2 ms, launch/sync overhead — the launch-diet half
of the campaign. The honest roofline remains ~86 tok/s (8B, measured
1290 GB/s read ceiling / 15.01 GB per token).

## [7] 70B TP=4 GEMV sweep + CK alignment fix (2026-10)

`gemv_70b_sweep.hip` (engine kernel verbatim, GCD 0, 50 iters) sweeps chunk
count on the 70B TP=4 shapes (qkv N=2560, o N=2048, gup N=14336, down
N=8192 — all K=8192 except down K=7168).

Findings:
1. **chunks=3/6 lose everywhere; 1/2/4/8 are fine.** The discriminator is CK
   alignment to the 256 B warp transaction: CK=2732/1368 (chunks 3/6) leaves
   chunk starts mid-stripe so every coalesced 256 B warp-read straddles two
   stripes. Rounding CK to 128 elements: gup chunks=3 772 -> 1017 GB/s
   (+32%), down chunks=6 929 -> 1173, qkv/o chunks=3 ~705 -> ~840-867.
   Landed in the engine (`gemv_launch`, kernels.hip) — 70B TP=4 decode
   51.6 -> 45.4 ms/tok end-to-end.
2. **Row-stride channel lockstep (secondary)**: K=8192 bf16 = 16 KB row
   stride ≡ 0 mod (32 channels × 256 B) — all rows start at the same HBM
   channel phase, warps march through channels in lockstep. down (K=7168,
   14 KB stride ≡ 24 mod 32) rotates phases and runs 1175-1231 GB/s (~95%
   ceiling) vs the 16 KB shapes' 775-1023. Padded-stride gup (16640 B/row)
   reaches ~1193 GB/s at chunks=1 — the next ~18% on that shape.
3. The engine's `gemv_chunks` policy (CTA target 4096, swept on 8B shapes)
   picks poorly for 70B shapes: gup got the worst case (chunks=3
   unaligned); qkv/o would rather run chunks=2 (1073/1024) than today's
   chunks=8 (853/847). Per-shape best from the sweep: qkv 2, o 2,
   gup 3-aligned, down 4.
