#pragma once

// GPU-resident SIMT kernels (design §3): trunk/state ops (rmsnorm, residual
// add, embed gather) plus the glue device blocks need around vendor GEMMs
// (dtype conversion, silu-mul, RoPE+KV-append, online-softmax attention,
// argmax). Plain SIMT only — no matrix intrinsics. Compiled by hipcc from
// kernels.hip; this header is the g++-safe host side (raw pointers only).
//
// bf16 buffers are passed as void* (truncating conversion, bit-identical to
// host bf16::from_f32). Everything launches on the default stream of the
// current device; callers set the device (single-GPU migration step: always
// device 0 via the platform singleton).

#include <cstddef>
#include <cstdint>

namespace alice::gpu::kernels {

// dst[i] = bf16(src[i]) (truncation). n elements.
void f32_to_bf16(const float* src, void* dst, size_t n);

// act[s*I+i] = bf16(silu(gu[s*2I+i]) * gu[s*2I+I+i]); gu is the fused
// gate/up GEMM output (col-major [2I, S] == row-major [S, 2I]);
// act is row-major [S, I] bf16.
void silu_mul(const float* gu, void* act, uint32_t S, uint32_t I);

// out[s][i] = in[s][i] * rsqrt(mean(in[s][:]^2) + eps) * gamma[i].
// out-of-place (Trunk read: residual -> io).
void rmsnorm(const float* in, const float* gamma, float* out, uint32_t S, uint32_t d,
             float eps);

// dst[i] += src[i] (Trunk write: residual += io).
void add_rows(float* dst, const float* src, size_t n);

// out[s][:] = fp32(table[ids[s]][:]); table is bf16 [vocab, d].
void embed_gather(const void* table, const int32_t* ids, float* out, uint32_t S,
                  uint32_t d);

// RoPE-rotate q (in place) and k at absolute positions [pos0, pos0+S), then
// append k,v to the caches at those positions. qkv is the fused projection
// output row-major [S, (nh+2*nkv)*hd]; caches are [max_seq, nkv*hd] fp32.
// cos_t/sin_t are [max_seq, hd/2] device tables (HF/Llama convention).
void rope_append(float* qkv, float* kcache, float* vcache, const float* cos_t,
                 const float* sin_t, uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv,
                 uint32_t hd);

// Causal attention with online softmax. Query s at absolute position pos0+s
// attends cache rows [0, pos0+s]. GQA: q-head h reads kv-head h/(nh/nkv).
// Output written as bf16 row-major [S, nh*hd] (o_proj GEMM consumes bf16).
void attention(const float* qkv, const float* kcache, const float* vcache, void* attn_bf16,
               uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv, uint32_t hd,
               float scale);

// *out = index of the largest element (ties: lowest index). Single block.
void argmax(const float* x, size_t n, int32_t* out);

}  // namespace alice::gpu::kernels
