#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "alice/tensor.hpp"

namespace alice {

// RMSNorm + plain add-residual trunk (design §7.2: "where the mess lives").
// Owns the residual-stream buffer, the single static io buffer, and all norm
// weights. A model proposing a new residual scheme gets a new Trunk class.
class RmsNormAddTrunk {
 public:
  // pre_norms: one validated bf16 [1, d] view PER BLOCK (execution order);
  // final_norm likewise. All copied into private fp32 buffers; freed on return.
  RmsNormAddTrunk(std::vector<TensorView> pre_norms, TensorView final_norm, float eps,
                  uint32_t hidden, uint32_t max_prefill_chunk);

  // Starts a forward pass of S tokens; returns the residual view [S, d]
  // (fp32) for the Embedder to fill. S <= max_prefill_chunk (hard error).
  TensorView begin_pass(uint32_t S);

  TensorView io();  // current io view [S, d] fp32

  void read(size_t block);   // io = rmsnorm(residual, pre_norm[block])
  void write(size_t block);  // residual += io
  void output();             // io = rmsnorm(residual, final_norm)

  uint32_t hidden() const { return d_; }

 private:
  uint32_t d_ = 0, mpc_ = 0, S_ = 0;
  float eps_ = 0.0f;
  std::unique_ptr<float[]> residual_;   // [mpc, d] fp32
  std::unique_ptr<float[]> io_;         // [mpc, d] fp32
  std::vector<float> norms_;            // [nblocks][d] flattened, fp32
  std::vector<float> final_;            // [d] fp32
};

}  // namespace alice
