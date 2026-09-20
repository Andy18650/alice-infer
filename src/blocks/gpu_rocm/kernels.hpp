#pragma once

// gpu_rocm block-private SIMT kernels (design §3: optimized impls are
// self-contained — private math, no shared kernel toolbox). Glue the device
// blocks need around vendor GEMMs: dtype conversion, silu-mul,
// RoPE+KV-append, online-softmax attention. Plain SIMT only — no matrix
// intrinsics. Compiled by hipcc from kernels.hip; this header is the
// g++-safe host side (raw pointers only).
//
// bf16 buffers are passed as void* (truncating conversion, bit-identical to
// host bf16::from_f32). Everything launches on the default stream of the
// current device; callers set the device.

#include <cstddef>
#include <cstdint>

namespace alice::gpu_rocm::kernels {

// dst[i] = bf16(src[i]) (truncation). n elements.
void f32_to_bf16(const float* src, void* dst, size_t n);

// act[s*I+i] = bf16(silu(gu[s*2I+i]) * gu[s*2I+I+i]); gu is the fused
// gate/up GEMM output (col-major [2I, S] == row-major [S, 2I]);
// act is row-major [S, I] bf16.
void silu_mul(const float* gu, void* act, uint32_t S, uint32_t I);

// RoPE-rotate q (in place) and k at absolute positions [pos0, pos0+S), then
// append k,v to the caches at those positions. qkv is the fused projection
// output row-major [S, (nh+2*nkv)*hd]; caches are [max_seq, nkv*hd] fp32.
// cos_t/sin_t are [max_seq, hd/2] device tables (HF/Llama convention).
void rope_append(float* qkv, float* kcache, float* vcache, const float* cos_t,
                 const float* sin_t, uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv,
                 uint32_t hd);

// Attention dispatch (all measured in research/prefill-attn-ab; v1 retired):
//   S > kAttnSplitMaxS  -> v4 split-T prefill: one block per (query-tile,
//     key-chunk, q-head), thread-per-query, raw scores in a key-major LDS
//     tile, deferred exp with the single-chunk max, partials to a SHARED
//     scratch (loader-allocated: nh * max_prefill_chunk * ceil(max_seq/64) *
//     (hd+2) floats), merged by a combine kernel. hd must be 64 or 128.
//   S <= kAttnSplitMaxS -> v2 flash-decode (all T): one block per (split,
//     query, head), 4 warps striding rows with in-warp reductions; partials
//     merged by a combine kernel. split_scratch must hold kAttnMaxSplits *
//     nh * kAttnSplitMaxS * (hd + 2) floats (block-allocated).
inline constexpr uint32_t kAttnSplitMaxS = 4;
inline constexpr uint32_t kAttnRowsPerSplit = 64;
inline constexpr uint32_t kAttnMaxSplits = 64;
inline constexpr uint32_t kPrefillChunkKeys = 64;

// --- S=1 GEMV (decode dispatch; design §15, measured in
// research/decode-roofline) --------------------------------------------------
//
// At S==1 rocBLAS gemm_ex picks no-split-K GEMM kernels that leave most of
// the HBM bandwidth on the table on the FFN shapes (down [4096x14336] at
// 313 GB/s, gup [28672x4096] at 727 GB/s, vs a measured 1290 GB/s pure-read
// ceiling; GEMMs were 26.0 of 29.6 ms/token). These family-private split-K
// GEMVs replace gemm_ex at S==1 in both blocks: warp-per-row over K-chunks
// (chunk count chosen by gemv_chunks to occupy the GCD), uint64 weight
// loads (4 bf16 per lane per step), partials + combine, fp32 accumulation
// over bf16 operands — same operand-quantization contract as the GEMM path.
//
// x is either fp32 (qkv/gup: the fp32 stream — operand quantization is
// fused in-register, bit-identical to the standalone f32_to_bf16 kernel,
// so the separate conversion launch disappears) or bf16 (o/down: the
// attention/activation outputs are already bf16). K must be a multiple of
// 4 (uint64/float4 loads); N has no constraint. Fixed schedule per
// (N, K): deterministic (design §10).

// Split count for the S=1 GEMV: enough (row-block, chunk) blocks to occupy
// the GCD without splitting K below ~1024 per chunk. Single tuning knob;
// the target constant is measured (see the study).
uint32_t gemv_chunks(uint32_t N, uint32_t K);

// Partials scratch a GEMV site needs (floats; produced and consumed within
// one call — block-private allocation today, §14 scratch-pool candidate).
inline size_t gemv_partial_floats(uint32_t N, uint32_t K) {
  return size_t(gemv_chunks(N, K)) * N;
}

// y[N] (fp32) = W (bf16 [N,K] row-major) . x. partials: gemv_partial_floats(N,K)
// floats. K % 4 == 0 required (loud failure otherwise).

// Causal attention with online softmax. Query s at absolute position pos0+s
// attends cache rows [0, pos0+s]. GQA: q-head h reads kv-head h/(nh/nkv).
// Output written as bf16 row-major [S, nh*hd] (o_proj GEMM consumes bf16).
void attention(const float* qkv, const float* kcache, const float* vcache, void* attn_bf16,
               float* split_scratch, float* prefill_scratch, uint32_t pos0, uint32_t S,
               uint32_t nh, uint32_t nkv, uint32_t hd, float scale);

// y[N] (fp32) = W (bf16 [N,K] row-major) . x — see the S=1 GEMV block above.
// fp32 x: quantized in-register (bit-identical to f32_to_bf16).
void gemv_x_f32(const void* W, const float* x, float* y, float* partials, uint32_t N,
                uint32_t K);
// bf16 x: already-quantized operand (attention/activation output).
void gemv_x_bf16(const void* W, const void* x, float* y, float* partials, uint32_t N,
                 uint32_t K);

}  // namespace alice::gpu_rocm::kernels
