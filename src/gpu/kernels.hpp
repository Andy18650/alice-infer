#pragma once

// Engine trunk/state SIMT kernels (design §3): rmsnorm, residual add, embed
// gather, argmax, plus the dtype conversion the LmHead needs before its
// GEMM. Plain SIMT only — no matrix intrinsics; this is the set the future
// HIP/CUDA/MUSA spelling header will map. Block-private kernels live with
// their impl family (src/blocks/gpu_rocm/), never here. Compiled by hipcc
// from kernels.hip; this header is the g++-safe host side (raw pointers
// only).
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

// out[s][i] = in[s][i] * rsqrt(mean(in[s][:]^2) + eps) * gamma[i].
// out-of-place (Trunk read: residual -> io).
void rmsnorm(const float* in, const float* gamma, float* out, uint32_t S, uint32_t d,
             float eps);

// dst[i] += src[i] (Trunk write: residual += io).
void add_rows(float* dst, const float* src, size_t n);

// out[s][:] = fp32(table[ids[s]][:]); table is bf16 [vocab, d].
void embed_gather(const void* table, const int32_t* ids, float* out, uint32_t S,
                  uint32_t d);

// *out = index of the largest element (ties: lowest index). Single block.
void argmax(const float* x, size_t n, int32_t* out);

}  // namespace alice::gpu::kernels
