// Standalone kernels for the P2P allreduce study (research/p2p-allreduce/).
// hipcc-compiled; declared g++-safe in kernels.hpp.

#include "kernels.hpp"

#include <hip/hip_runtime.h>

namespace p2p_ar {

namespace {
unsigned grid_for(size_t n, unsigned block) {
  return unsigned((n + block - 1) / block);
}
}  // namespace

__global__ void k_fill(float* __restrict__ dst, uint32_t n, float v) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = v;
}

void fill_value(float* dst, uint32_t n, float v) {
  k_fill<<<grid_for(n, 256), 256>>>(dst, n, v);
}

__global__ void k_sum(const float* __restrict__ scratch, size_t slot_n,
                      float* __restrict__ dst, uint32_t n, uint32_t k) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = 0.0f;
  for (uint32_t r = 0; r < k; ++r) s += scratch[size_t(r) * slot_n + i];
  dst[i] = s;
}

void sum_reduce(const float* scratch, size_t slot_n, float* dst, uint32_t n,
                uint32_t k) {
  k_sum<<<grid_for(n, 256), 256>>>(scratch, slot_n, dst, n, k);
}

// Flat P2P allreduce: read phase (sum all peers' io into local scratch) and
// write phase (copy scratch -> own io). See kernels.hpp. peer_io is a device
// array of ws io pointers; peer j's io is remote for j != this rank (peer
// access). io must be fine-grained-allocated so the cross-GCD reads are
// coherent without consumer-L2 staleness across allreduces.
__global__ void k_flat_read(float* __restrict__ scratch,
                            float* const* __restrict__ peer_io, uint32_t n,
                            uint32_t ws) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = 0.0f;
  for (uint32_t j = 0; j < ws; ++j) s += peer_io[j][i];
  scratch[i] = s;
}

__global__ void k_flat_write(float* __restrict__ dst,
                             const float* __restrict__ src, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

void flat_read(float* scratch, float* const* peer_io, uint32_t n, uint32_t ws) {
  k_flat_read<<<grid_for(n, 256), 256>>>(scratch, peer_io, n, ws);
}

// Fused out-of-place: dst[i] = sum_j peer_io[j][i]. One kernel (no separate
// scratch + write), but the read→write barrier across blocks still applies.
__global__ void k_flat_read_write(float* __restrict__ dst,
                                 float* const* __restrict__ peer_io,
                                 uint32_t n, uint32_t ws) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = 0.0f;
  for (uint32_t j = 0; j < ws; ++j) s += peer_io[j][i];
  dst[i] = s;
}

void flat_read_write(float* dst, float* const* peer_io, uint32_t n, uint32_t ws) {
  k_flat_read_write<<<grid_for(n, 256), 256>>>(dst, peer_io, n, ws);
}

void flat_write(float* dst, const float* src, uint32_t n) {
  k_flat_write<<<grid_for(n, 256), 256>>>(dst, src, n);
}

}  // namespace p2p_ar
