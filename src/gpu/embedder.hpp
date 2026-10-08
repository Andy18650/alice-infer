#pragma once

// GPU-resident Embedder: bf16 table on device, gather into the (device)
// residual stream. Token ids are the model's only regular H2D traffic.

#include <cstdint>
#include <span>

// hipStream_t forward declaration (matches hip_runtime_api.h).
typedef struct ihipStream_t* hipStream_t;

#include "alice/tensor.hpp"

namespace alice::gpu {

class Embedder {
 public:
  // table: staged bf16 [vocab, d] weights, moved in and freed on return.
  // stream: the rank's compute stream (Resources) — ids H2D + gather launch.
  Embedder(hipStream_t stream, Tensor<bf16> table, uint32_t max_prefill_chunk);
  ~Embedder();

  // residual += nothing: WRITES the gathered rows into residual [S, d] (fp32).
  void embed(std::span<const int32_t> ids, float* residual);

  uint32_t vocab_size() const { return vocab_; }

 private:
  hipStream_t stream_ = nullptr;
  uint32_t vocab_ = 0, d_ = 0, mpc_ = 0;
  void* table_ = nullptr;   // bf16 [vocab, d]
  int32_t* ids_dev_ = nullptr;  // [mpc]
};

}  // namespace alice::gpu
