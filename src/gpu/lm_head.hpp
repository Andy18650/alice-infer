#pragma once

// GPU-resident LmHead: bf16 weight on device, rocBLAS projection, logits
// stay on device (the Sampler consumes them there). project_last for the
// decode/step path; project_all for teacher-forced validation.

#include <cstdint>

#include "alice/tensor.hpp"

namespace alice::gpu {

class LmHead {
 public:
  // weight: validated bf16 [vocab, d] staging view, considered freed on return.
  LmHead(BF16View weight, uint32_t max_prefill_chunk);
  ~LmHead();

  // io: device fp32 [S, d] (the trunk's normed output).
  const float* project_last(const float* io, uint32_t S);  // -> logits [V]
  const float* project_all(const float* io, uint32_t S);   // -> logits [S, V]

  uint32_t vocab() const { return vocab_; }
  uint32_t hidden() const { return d_; }

 private:
  uint32_t vocab_ = 0, d_ = 0, mpc_ = 0;
  void* w_ = nullptr;        // bf16 [V, d]
  void* dx_ = nullptr;       // bf16 [mpc, d]
  float* logits_ = nullptr;  // fp32 [mpc, V] (row 0 reused by project_last)
};

}  // namespace alice::gpu
