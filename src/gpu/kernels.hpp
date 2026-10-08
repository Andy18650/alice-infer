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
// host bf16::from_f32). Callers select the device and explicitly pass its
// rank execution stream; tests may pass the default stream.

#include <cstddef>
#include <cstdint>

// Forward declaration (matches hip_runtime_api.h; keeps this header
// g++-safe — no HIP include). Every launcher takes the rank's compute
// stream as its first argument (gpu::Resources::stream()).
typedef struct ihipStream_t* hipStream_t;

namespace alice::gpu::kernels {

// dst[i] = bf16(src[i]) (truncation). n elements.
void f32_to_bf16(hipStream_t stream, const float* src, void* dst, size_t n);

// out[s][i] = in[s][i] * rsqrt(mean(in[s][:]^2) + eps) * gamma[i].
// out-of-place (Trunk read: residual -> io).
void rmsnorm(hipStream_t stream, const float* in, const float* gamma, float* out,
             uint32_t S, uint32_t d, float eps);

// dst[i] += src[i] (Trunk write: residual += io).
void add_rows(hipStream_t stream, float* dst, const float* src, size_t n);

// out[s][:] = fp32(table[ids[s]][:]); table is bf16 [vocab, d].
void embed_gather(hipStream_t stream, const void* table, const int32_t* ids,
                  float* out, uint32_t S, uint32_t d);

// *out = index of the largest element (ties: lowest index). Single block.
void argmax(hipStream_t stream, const float* x, size_t n, int32_t* out);

// --- Device-flag P2P allreduce (world_size > 1). ---------------------------
// Cross-GCD sync without host API calls: fine-grained device flags +
// system-scope atomic release/poll/acquire (research/p2p-allreduce
// atomics.md; probe-verified). Flags are 64-bit epoch counters (never wrap).

// Cross-GCD barrier at `value` (one block).
void ar_flagbar(hipStream_t stream, uint64_t* my_flag, uint64_t* const* flags,
                int ws, int rank, uint64_t value, const uint64_t* epoch_base = nullptr);

// out[i] = sum_j contribs[j][slot_offset + i] (float4, wide grid).
// out must not alias any published slot. n and slot_offset divisible by 4.
// Slots remain immutable until all ranks complete the forward.
void ar_gather_f4(hipStream_t stream, float* const* contribs, float* out, int ws,
                  uint32_t n, size_t slot_offset);

// Fused single-block out-of-place allreduce: produced barrier + gather.
// n divisible by 4 and <= 16384 floats (64 KB); slot_offset in floats,
// divisible by 4. vprod identifies this collective; no consumed epoch.
// During graph replay the live *epoch_base is added to the captured ordinal.
void ar_fused_f4(hipStream_t stream, float* out, float* const* contribs,
                 uint64_t* my_flag, uint64_t* const* flags, int ws, int rank,
                 uint32_t n, size_t slot_offset, uint64_t vprod,
                 const uint64_t* epoch_base = nullptr);

}  // namespace alice::gpu::kernels
