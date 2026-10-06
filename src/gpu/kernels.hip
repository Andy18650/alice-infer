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

void f32_to_bf16(const float* src, void* dst, size_t n) {
  k_f32_to_bf16<<<grid_for(n, 256), 256>>>(src, (unsigned short*)dst, n);
}

void rmsnorm(const float* in, const float* gamma, float* out, uint32_t S, uint32_t d,
             float eps) {
  k_rmsnorm<<<S, 256>>>(in, gamma, out, d, eps);
}

void add_rows(float* dst, const float* src, size_t n) {
  k_add_rows<<<grid_for(n, 256), 256>>>(dst, src, n);
}

void embed_gather(const void* table, const int32_t* ids, float* out, uint32_t S,
                  uint32_t d) {
  k_embed_gather<<<S, 256>>>((const unsigned short*)table, (const int*)ids, out, d);
}

void argmax(const float* x, size_t n, int32_t* out) {
  k_argmax<<<1, 1024>>>(x, n, (int*)out);
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
                             unsigned long long value) {
  if (threadIdx.x != 0) return;
  ar_flag_release(my_flag, value);
  for (int j = 0; j < ws; ++j)
    if (j != rank) while (ar_flag_poll(flags[j]) < value) {
    }
  ar_flag_acquire(my_flag);
}

// Wide read-gather: out[i] = sum_j contribs[j][i], float4. Writes a SEPARATE
// out buffer (the pipe-path scratch), so no in-place hazard.
__global__ void k_ar_gather_f4(float* const* contribs, float* __restrict__ out,
                               int ws, uint32_t n) {
  const uint32_t n4 = n / 4;
  float4* o4 = (float4*)out;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4;
       i4 += stride) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < ws; ++j) {
      const float4 v = ((const float4*)contribs[j])[i4];
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    o4[i4] = s;
  }
}

// Fused single-block allreduce (latency regime, n <= kArFusedMaxFloats):
// produce barrier -> read all ranks' contribs into REGISTERS -> consume
// barrier -> write io in place. The register hold across the consume barrier
// is what makes the in-place write safe: peers are still reading my io during
// my gather; my overwrite waits until every rank has finished reading
// (consume flag), and my sums sit in registers in between. Sum order j =
// 0..ws-1 on every rank => bitwise-identical results everywhere.
constexpr uint32_t kArMaxHoldF4 = 4;  // = kArFusedMaxFloats/4/1024
__global__ void k_ar_fused_f4(float* io, float* const* contribs,
                              unsigned long long* my_flag,
                              unsigned long long* const* flags, int ws,
                              int rank, uint32_t n, unsigned long long vprod,
                              unsigned long long vcons) {
  const uint32_t n4 = n / 4;
  if (threadIdx.x == 0) {
    ar_flag_release(my_flag, vprod);
    for (int j = 0; j < ws; ++j)
      if (j != rank) while (ar_flag_poll(flags[j]) < vprod) {
      }
    ar_flag_acquire(my_flag);
  }
  __syncthreads();
  float4 hold[kArMaxHoldF4];
#pragma unroll
  for (uint32_t u = 0; u < kArMaxHoldF4; ++u) {
    const uint32_t i4 = threadIdx.x + u * blockDim.x;
    if (i4 >= n4) break;
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < ws; ++j) {
      const float4 v = ((const float4*)contribs[j])[i4];
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    hold[u] = s;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ar_flag_release(my_flag, vcons);
    for (int j = 0; j < ws; ++j)
      if (j != rank) while (ar_flag_poll(flags[j]) < vcons) {
      }
    ar_flag_acquire(my_flag);
  }
  __syncthreads();
  float4* o4 = (float4*)io;
#pragma unroll
  for (uint32_t u = 0; u < kArMaxHoldF4; ++u) {
    const uint32_t i4 = threadIdx.x + u * blockDim.x;
    if (i4 >= n4) break;
    o4[i4] = hold[u];
  }
}

void ar_flagbar(uint64_t* my_flag, uint64_t* const* flags, int ws, int rank,
                uint64_t value) {
  k_ar_flagbar<<<1, 32>>>((unsigned long long*)my_flag,
                          (unsigned long long* const*)flags, ws, rank,
                          (unsigned long long)value);
}

void ar_gather_f4(float* const* contribs, float* out, int ws, uint32_t n) {
  uint32_t grid = (n / 4 + 255) / 256;
  if (grid > 220) grid = 220;
  if (grid < 1) grid = 1;
  k_ar_gather_f4<<<grid, 256>>>(contribs, out, ws, n);
}

void ar_fused_f4(float* io, float* const* contribs, uint64_t* my_flag,
                 uint64_t* const* flags, int ws, int rank, uint32_t n,
                 uint64_t vprod, uint64_t vcons) {
  k_ar_fused_f4<<<1, 1024>>>(io, contribs, (unsigned long long*)my_flag,
                             (unsigned long long* const*)flags, ws, rank, n,
                             (unsigned long long)vprod,
                             (unsigned long long)vcons);
}

}  // namespace alice::gpu::kernels
