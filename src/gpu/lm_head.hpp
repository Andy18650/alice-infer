#pragma once

// GPU-resident LmHead: bf16 weight on device, rocBLAS projection, logits
// stay on device until Model downloads them for host sampling. project_last for the
// decode/step path; project_all for teacher-forced validation.

#include <cstdint>

// hipStream_t forward declaration (matches hip_runtime_api.h).
typedef struct ihipStream_t* hipStream_t;

#include "blas.hpp"
#include "alice/tensor.hpp"

namespace alice::gpu {

class LmHead {
 public:
  // weight: staged bf16 [vocab, d] weights, moved in and freed on return.
  // stream: the rank's compute stream (Resources; BLAS handle is bound to
  // the same stream).
  LmHead(hipStream_t stream, Tensor<bf16> weight, uint32_t max_prefill_chunk,
         rocblas_handle handle);
  ~LmHead();

  // io: device fp32 [S, d] (the trunk's normed output).
  const float* project_last(const float* io, uint32_t S);  // -> logits [V]
  const float* project_all(const float* io, uint32_t S);   // -> logits [S, V]

  uint32_t vocab() const { return vocab_; }
  uint32_t hidden() const { return d_; }

 private:
  hipStream_t stream_ = nullptr;
  uint32_t vocab_ = 0, d_ = 0, mpc_ = 0;
  void* w_ = nullptr;        // bf16 [V, d]
  void* dx_ = nullptr;       // bf16 [mpc, d]
  float* logits_ = nullptr;  // fp32 [mpc, V] (row 0 reused by project_last)
  rocblas_handle handle_ = nullptr;  // borrowed from the Model's Resources
};

}  // namespace alice::gpu
