// Engine trunk/state SIMT kernels (design §3). Compiled by hipcc (see
// CMakeLists alice_gpu_kernels.o). Plain SIMT: shared-memory reductions
// only, no matrix intrinsics. bf16 conversions truncate (__float_as_uint >>
// 16), bit-identical to host bf16::from_f32.

#include <hip/hip_runtime.h>

#include "kernels.hpp"

namespace {

__device__ __forceinline__ unsigned short f2b(float f) {
  return (unsigned short)(__float_as_uint(f) >> 16);
}
__device__ __forceinline__ float b2f(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// Block-wide sum over NT threads via shared memory (no shuffle tricks).
template <unsigned NT>
__device__ float block_reduce_sum(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  return shared[0];
}

__global__ void k_f32_to_bf16(const float* __restrict__ src, unsigned short* __restrict__ dst,
                              size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t k = i; k < n; k += stride) dst[k] = f2b(src[k]);
}

__global__ void k_rmsnorm(const float* __restrict__ in, const float* __restrict__ gamma,
                          float* __restrict__ out, uint32_t d, float eps) {
  __shared__ float shared[256];
  const uint32_t s = blockIdx.x;
  const float* row = in + (size_t)s * d;
  float* orow = out + (size_t)s * d;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += row[i] * row[i];
  ss = block_reduce_sum<256>(ss, shared);
  const float inv = 1.0f / sqrtf(ss / (float)d + eps);
  for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) orow[i] = row[i] * inv * gamma[i];
}

__global__ void k_add_rows(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t k = i; k < n; k += stride) dst[k] += src[k];
}

__global__ void k_embed_gather(const unsigned short* __restrict__ table,
                               const int* __restrict__ ids, float* __restrict__ out,
                               uint32_t d) {
  const uint32_t s = blockIdx.x;
  const unsigned short* row = table + (size_t)ids[s] * d;
  for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) out[(size_t)s * d + i] = b2f(row[i]);
}

__global__ void k_argmax(const float* __restrict__ x, size_t n, int* __restrict__ out) {
  __shared__ float vals[1024];
  __shared__ int idxs[1024];
  float best = -INFINITY;
  int besti = 0;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (x[i] > best) {
      best = x[i];
      besti = (int)i;
    }
  }
  vals[threadIdx.x] = best;
  idxs[threadIdx.x] = besti;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = vals[threadIdx.x + s];
      const int oi = idxs[threadIdx.x + s];
      if (ov > vals[threadIdx.x] || (ov == vals[threadIdx.x] && oi < idxs[threadIdx.x])) {
        vals[threadIdx.x] = ov;
        idxs[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = idxs[0];
}

unsigned grid_for(size_t n, unsigned block) {
  unsigned g = (unsigned)((n + block - 1) / block);
  return g == 0 ? 1 : (g > 65535 ? 65535 : g);
}

}  // namespace

namespace alice::gpu::kernels {

void f32_to_bf16(hipStream_t stream, const float* src, void* dst, size_t n) {
  k_f32_to_bf16<<<grid_for(n, 256), 256, 0, stream>>>(src, (unsigned short*)dst, n);
}

void rmsnorm(hipStream_t stream, const float* in, const float* gamma, float* out,
             uint32_t S, uint32_t d, float eps) {
  k_rmsnorm<<<S, 256, 0, stream>>>(in, gamma, out, d, eps);
}

void add_rows(hipStream_t stream, float* dst, const float* src, size_t n) {
  k_add_rows<<<grid_for(n, 256), 256, 0, stream>>>(dst, src, n);
}

void embed_gather(hipStream_t stream, const void* table, const int32_t* ids,
                  float* out, uint32_t S, uint32_t d) {
  k_embed_gather<<<S, 256, 0, stream>>>((const unsigned short*)table, (const int*)ids, out, d);
}

void argmax(hipStream_t stream, const float* x, size_t n, int32_t* out) {
  k_argmax<<<1, 1024, 0, stream>>>(x, n, (int*)out);
}

// ---------------------------------------------------------------------------
// Device-flag P2P allreduce (research/p2p-allreduce, atomics.md). Cross-GCD
// synchronization WITHOUT host API calls: fine-grained device-memory flags +
// system-scope atomic release/acquire. Writer: RELEASE/SYSTEM flag store
// (emits buffer_wbl2 — L2 writeback to HBM; load-bearing because XGMI reads
// are served by HBM, not the owner's L2). Reader: RELAXED/SYSTEM glc-poll of
// peers' flags, then ONE ACQUIRE/SYSTEM load (emits buffer_invl2, broadcast
// to all L2 slices + L1 writeback-invalidate) before touching data. This is
// the probe-verified protocol (atomicprobe/flagprobe: 0 mismatches, ws=2-8).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ar_flag_release(unsigned long long* f,
                                                unsigned long long v) {
  __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned long long ar_flag_poll(
    const unsigned long long* f) {
  return __hip_atomic_load((unsigned long long*)f, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void ar_flag_acquire(const unsigned long long* f) {
  (void)__hip_atomic_load((unsigned long long*)f, __ATOMIC_ACQUIRE,
                          __HIP_MEMORY_SCOPE_SYSTEM);
}

// One-block cross-GCD barrier at `value` (epoch counters never repeat).
__global__ void k_ar_flagbar(unsigned long long* my_flag,
                             unsigned long long* const* flags, int ws, int rank,
                             unsigned long long value,
                             const unsigned long long* epoch_base) {
  if (threadIdx.x != 0) return;
  if (epoch_base) value += *epoch_base;
  ar_flag_release(my_flag, value);
  for (int j = 0; j < ws; ++j)
    if (j != rank) while (ar_flag_poll(flags[j]) < value) {
    }
  // gfx90a cache maintenance: own-flag acquire broadcasts L2 invalidation.
  // Relaxed peer polling + this invalidate is not a portable C++ HB proof.
  ar_flag_acquire(my_flag);
}

// Decode winner from standalone traversal research: source pointers hoisted
// once into 64 B LDS, contiguous chunks, cyclic dephasing. No payload prefetch
// or partial-output storage. FP32 order changes; canonical tolerance gates
// validate this implementation, not a claim of bitwise-identical replicas.
__global__ void k_ar_decode_gather_f4(float* const* contribs, float* __restrict__ out,
                                     int ws, int rank, uint32_t n, size_t offset) {
  __shared__ const float4* peers[8];
  if (threadIdx.x < static_cast<uint32_t>(ws))
    peers[threadIdx.x] = reinterpret_cast<const float4*>(contribs[threadIdx.x] + offset);
  __syncthreads();
  const uint32_t n4 = n / 4;
  const uint32_t span = ((n4 + gridDim.x - 1) / gridDim.x + 63) / 64 * 64;
  const uint32_t chunk = (blockIdx.x + rank) % gridDim.x;
  const size_t start = size_t(chunk) * span;
  const size_t end = min(size_t(n4), start + span);
  const int phase = int((uint32_t(rank) + blockIdx.x) % uint32_t(ws));
  for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    int peer = phase;
    for (int step = 0; step < ws; ++step) {
      const float4 v = peers[peer][i];
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
      if (++peer == ws) peer = 0;
    }
    reinterpret_cast<float4*>(out)[i] = sum;
  }
}

// Existing single-CTA path for tiny rows: avoids the extra launch and LDS
// setup when a row does not benefit from a second CU's memory pipeline.
__global__ void k_ar_fused_f4(float* out, float* const* contribs,
                              unsigned long long* my_flag,
                              unsigned long long* const* flags, int ws, int rank,
                              uint32_t n, size_t offset, unsigned long long value,
                              const unsigned long long* epoch_base) {
  if (threadIdx.x == 0) {
    if (epoch_base) value += *epoch_base;
    ar_flag_release(my_flag, value);
    for (int j = 0; j < ws; ++j)
      if (j != rank) while (ar_flag_poll(flags[j]) < value) {}
    ar_flag_acquire(my_flag);
  }
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < n / 4; i += blockDim.x) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < ws; ++j) {
      const float4 v = reinterpret_cast<const float4*>(contribs[j] + offset)[i];
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    reinterpret_cast<float4*>(out)[i] = sum;
  }
}

void ar_flagbar(hipStream_t stream, uint64_t* my_flag, uint64_t* const* flags,
                int ws, int rank, uint64_t value, const uint64_t* epoch_base) {
  k_ar_flagbar<<<1, 32, 0, stream>>>((unsigned long long*)my_flag,
                          (unsigned long long* const*)flags, ws, rank,
                          (unsigned long long)value,
                          (const unsigned long long*)epoch_base);
}

void ar_decode_gather_f4(hipStream_t stream, float* const* contribs, float* out,
                          int ws, int rank, uint32_t n, size_t slot_offset) {
  // Measured 2x1024 at 70B's 32 KiB decode row. Model keeps tiny rows on
  // the fused path rather than extrapolating this launch to other shapes.
  k_ar_decode_gather_f4<<<2, 1024, 0, stream>>>(contribs, out, ws, rank, n, slot_offset);
}

void ar_fused_f4(hipStream_t stream, float* out, float* const* contribs,
                 uint64_t* my_flag, uint64_t* const* flags, int ws, int rank,
                 uint32_t n, size_t offset, uint64_t ordinal, const uint64_t* epoch_base) {
  k_ar_fused_f4<<<1, 1024, 0, stream>>>(out, contribs,
      reinterpret_cast<unsigned long long*>(my_flag),
      reinterpret_cast<unsigned long long* const*>(flags), ws, rank, n, offset,
      ordinal, reinterpret_cast<const unsigned long long*>(epoch_base));
}

}  // namespace alice::gpu::kernels
