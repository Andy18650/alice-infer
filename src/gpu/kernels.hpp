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

// --- Device-flag P2P allreduce (world_size > 1). ---------------------------
// Cross-GCD sync without host API calls: fine-grained device flags +
// system-scope atomic release/poll/acquire (research/p2p-allreduce
// atomics.md; probe-verified). Flags are 64-bit epoch counters (never wrap).

// Cross-GCD barrier at `value` (one block).
void ar_flagbar(uint64_t* my_flag, uint64_t* const* flags, int ws, int rank,
                uint64_t value);

// out[i] = sum_j contribs[j][i] (float4, wide grid; out must be a separate
// buffer, not any of the contribs). n divisible by 4.
void ar_gather_f4(float* const* contribs, float* out, int ws, uint32_t n);

// Fused single-block in-place allreduce (latency regime): produce barrier,
// gather all ranks' contribs into registers, consume barrier, write io.
// n divisible by 4 and <= 16384 floats (64 KB; 4 float4/thread at 1024
// threads). vprod/vcons = this call's produce/consume epoch values.
void ar_fused_f4(float* io, float* const* contribs, uint64_t* my_flag,
                 uint64_t* const* flags, int ws, int rank, uint32_t n,
                 uint64_t vprod, uint64_t vcons);

}  // namespace alice::gpu::kernels
