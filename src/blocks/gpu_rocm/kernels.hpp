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

// Forward declaration (matches hip_runtime_api.h; keeps this header
// g++-safe). Every launcher takes the rank's compute stream first
// (gpu::Resources::stream()).
typedef struct ihipStream_t* hipStream_t;

namespace alice::gpu_rocm::kernels {

// dst[i] = bf16(src[i]) (truncation). n elements.
void f32_to_bf16(hipStream_t stream, const float* src, void* dst, size_t n);

// act[s*I+i] = bf16(silu(gu[s*2I+i]) * gu[s*2I+I+i]); gu is the fused
// gate/up GEMM output (col-major [2I, S] == row-major [S, 2I]);
// act is row-major [S, I] bf16.
void silu_mul(hipStream_t stream, const float* gu, void* act, uint32_t S, uint32_t I);

// RoPE-rotate q (in place) and k at absolute positions [pos0, pos0+S), then
// append k,v to the caches at those positions. qkv is the fused projection
// output row-major [S, (nh+2*nkv)*hd]; caches are [max_seq, nkv*hd] fp32.
// cos_t/sin_t are [max_seq, hd/2] device tables (HF/Llama convention).
void rope_append(hipStream_t stream, float* qkv, float* kcache, float* vcache,
                  const float* cos_t,
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
// ceiling; GEMMs were 26.0 of 29.6 ms/token). The family-private split-K
// GEMVs replace gemm_ex at S==1 in both blocks. 2026-10 revision
// (research/decode-roofline [8-10]): the warp-per-row split-K was retired
// for a COLUMN-MAJOR thread-per-row-pair GEMV — weights are repacked at
// load (repack_to_col_major) so one thread loads a uint32 covering two
// adjacent rows of one column; a wave's 64 lanes then read a contiguous
// 256 B per step (fully coalesced), and K is split into 256-element chunks
// so thread count = (N/2)*(K/256) keeps the GCD occupied (gup's 14336
// threads were half the 28160-thread fill). Measured: gup 1277 GB/s (99%
// of the read ceiling) vs 1017 for the retired warp-per-row; fp32
// accumulation over bf16 operands — same operand-quantization contract as
// the GEMM path. The S>1 gemm_ex path consumes the same column-major
// layout with op=N, lda=N.
//
// x is either fp32 (qkv/gup: the fp32 stream — operand quantization is
// fused in-register, bit-identical to the standalone f32_to_bf16 kernel,
// so the separate conversion launch disappears) or bf16 (o/down: the
// attention/activation outputs are already bf16). K % 4 == 0 and N even
// required. Fixed schedule per (N, K): deterministic (design §10).

// Chunk count for the S=1 GEMV: 256-element K-chunks (fixed; the sweep
// showed bandwidth tracks thread count until VRAM saturation).
uint32_t gemv_cm_chunks(uint32_t K);

// Partials scratch a GEMV site needs (floats; produced and consumed within
// one call — block-private allocation today, §14 scratch-pool candidate).
inline size_t gemv_partial_floats(uint32_t N, uint32_t K) {
  return size_t(gemv_cm_chunks(K)) * N;
}

// y[N] (fp32) = W (bf16 [N,K] row-major) . x. partials: gemv_partial_floats(N,K)
// floats. K % 4 == 0 required (loud failure otherwise).

// Causal attention with online softmax. Query s at absolute position pos0+s
// attends cache rows [0, pos0+s]. GQA: q-head h reads kv-head h/(nh/nkv).
// Output written as bf16 row-major [S, nh*hd] (o_proj GEMM consumes bf16).
void attention(hipStream_t stream, const float* qkv, const float* kcache,
                const float* vcache, void* attn_bf16,
               float* split_scratch, float* prefill_scratch, uint32_t pos0, uint32_t S,
               uint32_t nh, uint32_t nkv, uint32_t hd, float scale);

// y[N] (fp32) = W (bf16 [N,K] COLUMN-major — see the S=1 GEMV block
// above) . x. partials: gemv_partial_floats(N,K) floats.
// fp32 x: quantized in-register (bit-identical to f32_to_bf16).
void gemv_cm_x_f32(hipStream_t stream, const void* W, const float* x, float* y,
                    float* partials, uint32_t N,
                   uint32_t K);
// bf16 x: already-quantized operand (attention/activation output).
void gemv_cm_x_bf16(hipStream_t stream, const void* W, const void* x, float* y,
                     float* partials, uint32_t N,
                    uint32_t K);

// Fused gup decode path (one launch saved + the fp32 dgu_ round-trip
// eliminated): partial phase for the fp32-stream site, then a fused
// combine+SiLU writing bf16 act directly (act[i] = silu(g_i)*u_i, gate/up
// = rows [0,I)/[I,2I) of the fused gup partials). Bit-identical math to
// combine + silu_mul.
void gemv_cm_partial_x_f32(hipStream_t stream, const void* W, const float* x,
                            float* partials, uint32_t N,
                           uint32_t K);
void gemv_cm_combine_silu(hipStream_t stream, const float* partials, void* act_bf16,
                           uint32_t I, uint32_t K);

// Load-time weight repack: row-major [N,K] bf16 -> column-major [N,K] bf16
// (W_cm[k*N+n]), consumed by the S=1 GEMV and by the S>1 gemm_ex (op=N,
// lda=N). Async on the caller's stream (subsequent stream work is ordered).
void repack_to_col_major(hipStream_t stream, const void* W_rm, void* W_cm, uint32_t N,
                          uint32_t K);

}  // namespace alice::gpu_rocm::kernels
