#pragma once

// GPU-resident Embedder: bf16 table on device, gather into the (device)
// residual stream. Token ids are the model's only regular H2D traffic.

#include <cstdint>
#include <span>

#include "alice/tensor.hpp"

namespace alice::gpu {

class Embedder {
 public:
  // table: staged bf16 [vocab, d] weights, moved in and freed on return.
  Embedder(Tensor<bf16> table, uint32_t max_prefill_chunk);
  ~Embedder();

  // residual += nothing: WRITES the gathered rows into residual [S, d] (fp32).
  void embed(std::span<const int32_t> ids, float* residual);

  uint32_t vocab_size() const { return vocab_; }

 private:
  uint32_t vocab_ = 0, d_ = 0, mpc_ = 0;
  void* table_ = nullptr;   // bf16 [vocab, d]
  int32_t* ids_dev_ = nullptr;  // [mpc]
};

}  // namespace alice::gpu
