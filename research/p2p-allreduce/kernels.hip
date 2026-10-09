// Standalone gfx90a traversal study. No cooperative/election code. Bounded
// 2/4-load lookahead variants share the controls' publication and math order.
#include "kernels.hpp"
#include <hip/hip_runtime.h>

namespace p2p_ar {
namespace {
__device__ unsigned long long epoch(const Args& a) { return *a.epoch_base + a.ordinal; }
__device__ void ready(const Args& a) {
  const auto value = epoch(a);
  __hip_atomic_store(reinterpret_cast<unsigned long long*>(a.flag), value,
                     __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  for (int peer = 0; peer < a.ws; ++peer) {
    if (peer == a.rank) continue;
    while (__hip_atomic_load(reinterpret_cast<unsigned long long*>(a.flags[peer]),
                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value) {}
  }
  // gfx90a cache-maintenance idiom: invalidate reader L2 after readiness.
  // This own-flag acquire is not a portable peer release/acquire pairing.
  (void)__hip_atomic_load(reinterpret_cast<unsigned long long*>(a.flag),
                          __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float4 add(float4 sum, float4 v) {
  sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
  return sum;
}
__device__ __forceinline__ void gather_stride(const Args& a) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < a.n / 4; i += stride) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int peer = 0; peer < a.ws; ++peer)
      sum = add(sum, reinterpret_cast<const float4*>(a.peers[peer] + a.offset)[i]);
    reinterpret_cast<float4*>(a.out)[i] = sum;
  }
}

struct Range { uint32_t begin, end, phase; };
__device__ Range chunk_range(const Args& a) {
  // Whole-wave (64 float4) boundaries keep pointer selection wave-uniform.
  // Rotate CTA-to-chunk assignment and peer traversal by rank. Each CTA's
  // peer order is fixed, but different ranks may add in different FP32 orders;
  // these research variants need numerical validation before production.
  const uint32_t n4 = a.n / 4;
  const uint32_t span = ((n4 + gridDim.x - 1) / gridDim.x + 63) / 64 * 64;
  const uint32_t chunk = (blockIdx.x + a.rank) % gridDim.x;
  const size_t begin = size_t(chunk) * span;
  if (begin >= n4) return {0, 0, 0};
  return {uint32_t(begin), uint32_t(min(size_t(n4), begin + span)),
          (uint32_t(a.rank) + blockIdx.x) % uint32_t(a.ws)};
}
__device__ __forceinline__ void hoist(const Args& a, const float4** peers) {
  // Only addresses, not payloads, are hoisted. 64 B shared memory at ws=8;
  // no dynamically indexed private pointer array that might spill to scratch.
  if (threadIdx.x < a.ws)
    peers[threadIdx.x] = reinterpret_cast<const float4*>(a.peers[threadIdx.x] + a.offset);
  __syncthreads();
}

__device__ float input_value(int rank, unsigned long long value, uint32_t i) {
  return float((rank + 1) * 128 + (value % 1024) * 4 + (i % 31)) * 0.03125f;
}
__global__ void k_fill(Args a) {
  const auto value = epoch(a);
  const uint32_t stride = gridDim.x * blockDim.x;
  auto* dst = reinterpret_cast<float4*>(a.input + a.offset);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < a.n / 4; i += stride) {
    const uint32_t j = 4 * i;
    dst[i] = make_float4(input_value(a.rank, value, j), input_value(a.rank, value, j + 1),
                        input_value(a.rank, value, j + 2), input_value(a.rank, value, j + 3));
  }
}
__global__ void k_barrier(Args a) { if (threadIdx.x == 0) ready(a); }
__global__ void k_fused(Args a) {
  if (threadIdx.x == 0) ready(a);
  __syncthreads();
  gather_stride(a);
}
__global__ void k_stride(Args a) { gather_stride(a); }

// Control for the change from grid-stride to contiguous chunks. Same rank
// assignment as phase/peer-major, but canonical peer order and no hoisting.
__global__ void k_chunk(Args a) {
  const auto r = chunk_range(a);
  for (uint32_t i = r.begin + threadIdx.x; i < r.end; i += blockDim.x) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int peer = 0; peer < a.ws; ++peer)
      sum = add(sum, reinterpret_cast<const float4*>(a.peers[peer] + a.offset)[i]);
    reinterpret_cast<float4*>(a.out)[i] = sum;
  }
}

// data-major: finish one float4's sum, then process another float4.
template <bool Dephase>
__global__ void k_hoisted(Args a) {
  __shared__ const float4* peers[kMaxPeers];
  hoist(a, peers);
  const auto r = chunk_range(a);
  for (uint32_t i = r.begin + threadIdx.x; i < r.end; i += blockDim.x) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    int peer = Dephase ? int(r.phase) : 0;
    for (int step = 0; step < a.ws; ++step) {
      sum = add(sum, peers[peer][i]);
      if constexpr (Dephase) {
        if (++peer == a.ws) peer = 0;
      } else ++peer;
    }
    reinterpret_cast<float4*>(a.out)[i] = sum;
  }
}

// peer-major: stream this CTA's entire chunk from a peer before moving on.
// Each thread owns its output elements; no inter-peer CTA barrier is needed.
// Extra local output reads/writes are explicit, not hidden extra kernels.
__global__ void k_peer_major(Args a) {
  __shared__ const float4* peers[kMaxPeers];
  hoist(a, peers);
  const auto r = chunk_range(a);
  auto* out = reinterpret_cast<float4*>(a.out);
  int peer = int(r.phase);
  for (int step = 0; step < a.ws; ++step) {
    const float4* src = peers[peer];
    for (uint32_t i = r.begin + threadIdx.x; i < r.end; i += blockDim.x) {
      const float4 sum = step == 0 ? make_float4(0.f, 0.f, 0.f, 0.f) : out[i];
      out[i] = add(sum, src[i]);
    }
    if (++peer == a.ws) peer = 0;
  }
}
// Resolve all LDS address lookups before payload issue. This is only a
// compiler dependency barrier (no fence/invalidation instruction). Without
// it Clang can place lgkmcnt(0) between flat loads and collapse the lookahead.
template <int Depth>
__device__ __forceinline__ void settle_pointers(const float4* (&src)[Depth]) {
  if constexpr (Depth == 2)
    asm volatile("" : "+v"(src[0]), "+v"(src[1]) : : "memory");
  else
    asm volatile("" : "+v"(src[0]), "+v"(src[1]), "+v"(src[2]), "+v"(src[3]) : : "memory");
}

// Data-major lookahead: issue Depth peer loads before consuming the first.
// Compile-time arrays + unrolled indices stay in registers. Pointer lookup
// precedes payload issue; tail peers are scalar so ws=2 also supports Depth=4.
template <int Depth, bool Dephase>
__global__ void k_prefetch(Args a) {
  static_assert(Depth == 2 || Depth == 4);
  __shared__ const float4* peers[kMaxPeers];
  hoist(a, peers);
  const auto r = chunk_range(a);
#pragma clang loop unroll(disable)
  for (uint32_t i = r.begin + threadIdx.x; i < r.end; i += blockDim.x) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    int peer = Dephase ? int(r.phase) : 0;
    int step = 0;
    // Prevent automatic outer-loop unrolling from turning both variants
    // into an eight-peer batch. Depth must describe the emitted lookahead.
#pragma clang loop unroll(disable)
    for (; step + Depth <= a.ws; step += Depth) {
      const float4* src[Depth];
#pragma unroll
      for (int j = 0; j < Depth; ++j) {
        src[j] = peers[peer];
        if (++peer == a.ws) peer = 0;
      }
      settle_pointers(src);
      float4 value[Depth];
#pragma unroll
      for (int j = 0; j < Depth; ++j) value[j] = src[j][i];
#pragma unroll
      for (int j = 0; j < Depth; ++j) sum = add(sum, value[j]);
    }
    for (; step < a.ws; ++step) {
      sum = add(sum, peers[peer][i]);
      if (++peer == a.ws) peer = 0;
    }
    reinterpret_cast<float4*>(a.out)[i] = sum;
  }
}

// Peer-major lookahead: batch consecutive vectors assigned to this thread,
// all from the same peer. Local partial-output traffic is still explicit.
template <int Depth>
__global__ void k_peer_prefetch(Args a) {
  static_assert(Depth == 2 || Depth == 4);
  __shared__ const float4* peers[kMaxPeers];
  hoist(a, peers);
  const auto r = chunk_range(a);
  auto* out = reinterpret_cast<float4*>(a.out);
  int peer = int(r.phase);
#pragma clang loop unroll(disable)
  for (int step = 0; step < a.ws; ++step) {
    const float4* src = peers[peer];
    uint32_t i = r.begin + threadIdx.x;
#pragma clang loop unroll(disable)
    for (; i + (Depth - 1) * blockDim.x < r.end; i += Depth * blockDim.x) {
      float4 value[Depth];
#pragma unroll
      for (int j = 0; j < Depth; ++j) value[j] = src[i + j * blockDim.x];
#pragma unroll
      for (int j = 0; j < Depth; ++j) {
        const uint32_t index = i + j * blockDim.x;
        const float4 sum = step == 0 ? make_float4(0.f, 0.f, 0.f, 0.f) : out[index];
        out[index] = add(sum, value[j]);
      }
    }
    for (; i < r.end; i += blockDim.x) {
      const float4 sum = step == 0 ? make_float4(0.f, 0.f, 0.f, 0.f) : out[i];
      out[i] = add(sum, src[i]);
    }
    if (++peer == a.ws) peer = 0;
  }
}
// One tile's partials remain in registers through all peers. Full tiles have
// no per-vector/per-peer bounds checks; only the final ragged tile is masked.
template <int Vectors, bool Tail>
__device__ __forceinline__ void register_tile(const Args& a, const float4* const* peers,
                                             Range r, uint32_t base) {
  float4 sum[Vectors];
#pragma unroll
  for (int j = 0; j < Vectors; ++j) sum[j] = make_float4(0.f, 0.f, 0.f, 0.f);
  int peer = int(r.phase);
#pragma clang loop unroll(disable)
  for (int step = 0; step < a.ws; ++step) {
    const float4* src = peers[peer];
#pragma unroll
    for (int j = 0; j < Vectors; ++j) {
      const uint32_t index = base + j * blockDim.x;
      if (!Tail || index < r.end) sum[j] = add(sum[j], src[index]);
    }
    if (++peer == a.ws) peer = 0;
  }
  auto* out = reinterpret_cast<float4*>(a.out);
#pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    const uint32_t index = base + j * blockDim.x;
    if (!Tail || index < r.end) out[index] = sum[j];
  }
}

// Static 1/4/8/32-vector tiles. 32 uses up to 256 threads and a one-wave/EU
// launch bound to permit its register budget rather than spill for occupancy.
template <int Vectors>
__global__ void __launch_bounds__(Vectors == 32 ? 256 : 1024, 1)
    k_peer_register(Args a) {
  static_assert(Vectors == 1 || Vectors == 4 || Vectors == 8 || Vectors == 32);
  __shared__ const float4* peers[kMaxPeers];
  hoist(a, peers);
  const auto r = chunk_range(a);
  uint32_t base = r.begin + threadIdx.x;
#pragma clang loop unroll(disable)
  for (; base + (Vectors - 1) * blockDim.x < r.end; base += Vectors * blockDim.x)
    register_tile<Vectors, false>(a, peers, r, base);
  if (base < r.end) register_tile<Vectors, true>(a, peers, r, base);
}
}  // namespace

void fill(hipStream_t stream, const Args& args) {
  const int blocks = int((args.n / 4 + 255) / 256);
  k_fill<<<blocks > 220 ? 220 : blocks, 256, 0, stream>>>(args);
}
hipError_t producer_only(hipStream_t, Args, int, int) { return hipSuccess; }
hipError_t fused(hipStream_t stream, Args args, int, int) {
  k_fused<<<1, 1024, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t pipe(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_stride<<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t chunk(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_chunk<<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t hoisted(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_hoisted<false><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t phase(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_hoisted<true><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_major(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_major<<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t prefetch2(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_prefetch<2, true><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t prefetch4(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_prefetch<4, true><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t hoist_prefetch2(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_prefetch<2, false><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_prefetch2(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_prefetch<2><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_prefetch4(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_prefetch<4><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_reg1(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_register<1><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_reg4(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_register<4><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_reg8(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_register<8><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
hipError_t peer_reg32(hipStream_t stream, Args args, int blocks, int threads) {
  if (threads > 256) return hipErrorInvalidConfiguration;  // large-tile register budget
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_peer_register<32><<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}
}  // namespace p2p_ar
