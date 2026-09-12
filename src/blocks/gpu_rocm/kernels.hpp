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

// Causal attention with online softmax. Query s at absolute position pos0+s
// attends cache rows [0, pos0+s]. GQA: q-head h reads kv-head h/(nh/nkv).
// Output written as bf16 row-major [S, nh*hd] (o_proj GEMM consumes bf16).
void attention(const float* qkv, const float* kcache, const float* vcache, void* attn_bf16,
               uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv, uint32_t hd,
               float scale);

}  // namespace alice::gpu_rocm::kernels
