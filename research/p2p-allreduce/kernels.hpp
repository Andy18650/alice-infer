#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace p2p_ar {
inline constexpr int kMaxPeers = 8;

// Published input slots remain immutable until all ranks finish the batch.
struct Args {
  float* input = nullptr;
  float* out = nullptr;
  float* const* peers = nullptr;
  uint64_t* flag = nullptr;
  uint64_t* const* flags = nullptr;
  const uint64_t* epoch_base = nullptr;
  size_t offset = 0;
  uint64_t ordinal = 0;
  uint32_t n = 0;
  int rank = 0, ws = 0;
};

void fill(hipStream_t stream, const Args& args);
using Launch = hipError_t (*)(hipStream_t, Args, int blocks, int threads);
hipError_t producer_only(hipStream_t, Args, int, int);
hipError_t fused(hipStream_t, Args, int, int);  // reference, fixed 1x1024
hipError_t pipe(hipStream_t, Args, int blocks, int threads);  // original grid-stride
hipError_t chunk(hipStream_t, Args, int blocks, int threads); // contiguous chunk control
hipError_t hoisted(hipStream_t, Args, int blocks, int threads); // pointer-only control
hipError_t phase(hipStream_t, Args, int blocks, int threads); // dephasing + pointer hoist
hipError_t peer_major(hipStream_t, Args, int blocks, int threads); // flipped loop
hipError_t prefetch2(hipStream_t, Args, int blocks, int threads);  // adjacent peers
hipError_t prefetch4(hipStream_t, Args, int blocks, int threads);
hipError_t hoist_prefetch2(hipStream_t, Args, int blocks, int threads); // canonical order
hipError_t peer_prefetch2(hipStream_t, Args, int blocks, int threads); // vectors from one peer
hipError_t peer_prefetch4(hipStream_t, Args, int blocks, int threads);
hipError_t peer_reg1(hipStream_t, Args, int blocks, int threads);
hipError_t peer_reg4(hipStream_t, Args, int blocks, int threads);
hipError_t peer_reg8(hipStream_t, Args, int blocks, int threads);
hipError_t peer_reg32(hipStream_t, Args, int blocks, int threads);
}  // namespace p2p_ar
