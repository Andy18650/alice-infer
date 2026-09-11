#pragma once

// GPU-resident Trunk (design §8.2): owns the device stream buffers
// (residual, io), ALL norm weights (device fp32), and the norm/residual
// SIMT kernels. RMSNorm + add-residual (Llama family).

#include <cstdint>
#include <vector>

#include "alice/tensor.hpp"

namespace alice::gpu {

class RmsNormAddTrunk {
 public:
  RmsNormAddTrunk(std::vector<BF16View> pre_norms, BF16View final_norm, float eps,
                  uint32_t hidden, uint32_t max_prefill_chunk);
  ~RmsNormAddTrunk();

  float* begin_pass(uint32_t S);  // sizes the pass; returns residual
  float* residual() { return residual_; }
  float* io() { return io_; }
  uint32_t S() const { return S_; }

  void read(size_t block);   // io = rmsnorm(residual, gamma_block)
  void write(size_t block);  // residual += io
  void output();             // io = rmsnorm(residual, gamma_final)

 private:
  uint32_t d_ = 0, mpc_ = 0, S_ = 0;
  float eps_ = 0;
  float* residual_ = nullptr;
  float* io_ = nullptr;
  float* norms_ = nullptr;   // [n_blocks, d]
  float* final_ = nullptr;   // [d]
  size_t nblocks_ = 0;
};

}  // namespace alice::gpu
