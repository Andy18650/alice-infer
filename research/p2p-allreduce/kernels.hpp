#pragma once

// Standalone kernels for the P2P allreduce study (research/p2p-allreduce/).
// g++-safe host side (raw pointers only); kernels.hip is hipcc-compiled.

#include <cstddef>
#include <cstdint>

namespace p2p_ar {

// dst[:] = v  (the simulated per-rank "forward" output).
void fill_value(float* dst, uint32_t n, float v);

// dst[i] = sum_{r=0..k-1} scratch[r*slot_n + i].  The rank-0 reduce of the
// gathered peer io rows (see engine src/gpu/kernels.hip sum_reduce).
void sum_reduce(const float* scratch, size_t slot_n, float* dst, uint32_t n,
                uint32_t k);

// Flat P2P allreduce read phase: scratch[i] = sum_{j=0..ws-1} peer_io[j][i].
// peer_io is a device array of ws io pointers (peer j's io is remote for j !=
// this rank; reads go via peer access). io must be fine-grained-allocated
// (hipDeviceMallocFinegrained) so the cross-GCD reads are coherent without
// consumer-L2 staleness across allreduces.
void flat_read(float* scratch, float* const* peer_io, uint32_t n, uint32_t ws);

// Flat P2P allreduce, fused out-of-place: dst[i] = sum_{j=0..ws-1} peer_io[j][i].
// Same reads as flat_read but writes the sum straight to `dst` (a different
// buffer from the peers' io) — so no in-place read→write race within the
// kernel. (The read→write dependency still exists across blocks: the next
// forward overwrites the src; see the p2p-allreduce README.)
void flat_read_write(float* dst, float* const* peer_io, uint32_t n, uint32_t ws);

// Flat P2P allreduce write phase: dst[i] = src[i] (copy local scratch -> own io).
void flat_write(float* dst, const float* src, uint32_t n);

}  // namespace p2p_ar
