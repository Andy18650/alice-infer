// Self-contained gfx90a study. One gather body, explicit launch functions.
#include "kernels.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace p2p_ar {
namespace {

__device__ unsigned long long epoch(const Args& a) {
  return *a.epoch_base + a.ordinal;
}
__device__ void publish(const Args& a, unsigned long long value) {
  __hip_atomic_store(reinterpret_cast<unsigned long long*>(a.flag), value,
                     __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ unsigned long long poll(const uint64_t* flag) {
  return __hip_atomic_load(reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(flag)),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void acquire(const Args& a) {
  (void)__hip_atomic_load(reinterpret_cast<unsigned long long*>(a.flag),
                          __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void wait_peers(const Args& a, unsigned long long value) {
  for (int peer = 0; peer < a.ws; ++peer)
    if (peer != a.rank) while (poll(a.flags[peer]) < value) {}
}
__device__ __forceinline__ void gather(const Args& a) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < a.n / 4; i += stride) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int peer = 0; peer < a.ws; ++peer) {
      const float4 v = reinterpret_cast<const float4*>(a.peers[peer] + a.offset)[i];
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    reinterpret_cast<float4*>(a.out)[i] = sum;
  }
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
__global__ void k_gather(Args a) { gather(a); }

__global__ void k_barrier(Args a) {
  if (threadIdx.x != 0) return;
  const auto value = epoch(a);
  publish(a, value);
  wait_peers(a, value);
  acquire(a);
}

// Every CTA may publish: the preceding producer kernel has already completed
// on this rank's stream. Redundant releases/acquires are safe but can be costly.
__global__ void k_wide(Args a) {
  if (threadIdx.x == 0) {
    const auto value = epoch(a);
    publish(a, value);
    wait_peers(a, value);
    acquire(a);
  }
  __syncthreads();
  gather(a);
}

// Elect a currently resident CTA, never assume block 0 is scheduled first.
// A hard-coded block-0 leader plus oversubscribed spinning CTAs can deadlock.
// One release per rank; each CTA retains its own peer poll + system acquire.
__global__ void k_elected(Args a) {
  if (threadIdx.x == 0) {
    const auto value = epoch(a);
    const auto old = atomicExch(reinterpret_cast<unsigned long long*>(a.claim), value);
    if (old < value) publish(a, value);
    else while (poll(a.flag) < value) {}
    wait_peers(a, value);
    acquire(a);
  }
  __syncthreads();
  gather(a);
}

// Cooperative *single-device* grid barrier. Cross-device ordering still uses
// the proven P2P flags. All blocks are residency-limited and participate;
// even blocks with no payload must reach grid.sync(). No multi-device API.
__global__ void k_coop(Args a) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const auto value = epoch(a);
    publish(a, value);
    wait_peers(a, value);
    acquire(a);  // broadcast L2 invalidation; no payload read before grid sync
  }
  cooperative_groups::this_grid().sync();
  gather(a);
}

}  // namespace

void fill(hipStream_t stream, const Args& args) {
  const int blocks = int((args.n / 4 + 255) / 256);
  k_fill<<<blocks > 220 ? 220 : blocks, 256, 0, stream>>>(args);
}

hipError_t producer_only(hipStream_t, Args, int, int) { return hipSuccess; }

hipError_t barrier(hipStream_t stream, Args args, int, int) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  return hipGetLastError();
}

hipError_t fused(hipStream_t stream, Args args, int, int) {
  k_wide<<<1, 1024, 0, stream>>>(args);
  return hipGetLastError();
}

hipError_t pipe(hipStream_t stream, Args args, int blocks, int threads) {
  k_barrier<<<1, 32, 0, stream>>>(args);
  k_gather<<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}

hipError_t wide(hipStream_t stream, Args args, int blocks, int threads) {
  k_wide<<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}

hipError_t elected(hipStream_t stream, Args args, int blocks, int threads) {
  k_elected<<<blocks, threads, 0, stream>>>(args);
  return hipGetLastError();
}

hipError_t coop(hipStream_t stream, Args args, int blocks, int threads) {
  void* params[] = {&args};
  return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(k_coop),
                                    dim3(blocks), dim3(threads), params, 0, stream);
}

hipError_t coop_wide(hipStream_t stream, Args args, int blocks, int threads) {
  void* params[] = {&args};
  return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(k_wide),
                                    dim3(blocks), dim3(threads), params, 0, stream);
}

namespace {
hipError_t capacity(const void* kernel, int threads, int* blocks) {
  int device = 0, active = 0;
  auto e = hipGetDevice(&device);
  if (e != hipSuccess) return e;
  hipDeviceProp_t prop{};
  e = hipGetDeviceProperties(&prop, device);
  if (e != hipSuccess) return e;
  if (!prop.cooperativeLaunch) return hipErrorNotSupported;
  e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, 0);
  if (e == hipSuccess) *blocks = active * prop.multiProcessorCount;
  return e;
}
}  // namespace

hipError_t coop_capacity(int threads, int* blocks) {
  return capacity(reinterpret_cast<const void*>(k_coop), threads, blocks);
}
hipError_t coop_wide_capacity(int threads, int* blocks) {
  return capacity(reinterpret_cast<const void*>(k_wide), threads, blocks);
}

}  // namespace p2p_ar
