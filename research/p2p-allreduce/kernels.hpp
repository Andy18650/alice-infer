#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace p2p_ar {

// Inputs are immutable after their producer until all ranks finish the batch.
// Output never aliases an input. Epoch base changes only between batches.
struct Args {
  float* input = nullptr;
  float* out = nullptr;
  float* const* peers = nullptr;
  uint64_t* flag = nullptr;
  uint64_t* const* flags = nullptr;
  uint64_t* claim = nullptr;  // local CTA election, not a peer signal
  const uint64_t* epoch_base = nullptr;
  size_t offset = 0;         // floats from each slab base
  uint64_t ordinal = 0;      // 1..slots, plus the live epoch base
  uint32_t n = 0;
  int rank = 0, ws = 0;
};

// All code lives in this study. No production headers, objects, or libraries.
void fill(hipStream_t stream, const Args& args);
using Launch = hipError_t (*)(hipStream_t, Args, int blocks, int threads);
using Capacity = hipError_t (*)(int threads, int* blocks);

hipError_t producer_only(hipStream_t, Args, int, int);  // fill control: no reduction
hipError_t barrier(hipStream_t, Args, int, int);        // ready barrier only
hipError_t fused(hipStream_t, Args, int, int);          // one CTA, 1024 threads
hipError_t pipe(hipStream_t, Args, int blocks, int threads);  // barrier -> gather
hipError_t wide(hipStream_t, Args, int blocks, int threads);  // per-CTA barrier + gather
hipError_t elected(hipStream_t, Args, int blocks, int threads);
hipError_t coop(hipStream_t, Args, int blocks, int threads);  // leader + grid.sync + gather
hipError_t coop_wide(hipStream_t, Args, int blocks, int threads);  // cooperative launch control

// Query after peer enable; cooperative grids cannot exceed residency.
hipError_t coop_capacity(int threads, int* blocks);
hipError_t coop_wide_capacity(int threads, int* blocks);

}  // namespace p2p_ar
