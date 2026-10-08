#pragma once

// GPU-resident dense FFN block (design §8.1; gfx90a implementation:
// rocBLAS gemm_ex bf16→fp32 + SIMT silu-mul). Device-io successor of the
// Device-resident dense FFN: same math as the host reference (fused
// silu-mul, down GEMM into io), no host staging — the stream never leaves
// the device.

#include "../block.hpp"

#include "../../gpu/resources.hpp"

#include "alice/engine_config.hpp"
#include "../dense_ffn_config.hpp"
#include "alice/tensor.hpp"

namespace alice::gpu_rocm {

class DenseFfnBF16 final : public block::Block {
 public:
  // Staged weights are moved in and freed on return (fused/copied to the
  // device in the constructor). Device VRAM is allocated through `res`
  // (bookkeeping); the block borrows the handle + carved pointers and does
  // NOT free them (Resources owns them, freed at teardown).
  DenseFfnBF16(TensorView<bf16> w_gate, TensorView<bf16> w_up, TensorView<bf16> w_down,
              const block::DenseFfnConfig& cfg, const ModelConfig& mc,
              gpu::Resources& res);
  ~DenseFfnBF16() override;

  void forward(float* io, uint32_t S) override;

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;
  void* dw_gup_ = nullptr;   // bf16 [2I, d] (fused [Wg;Wu])
  void* dw_down_ = nullptr;  // bf16 [d, I]
  void* dx_ = nullptr;       // bf16 [mpc, d]
  float* dgu_ = nullptr;     // fp32 [mpc, 2I] (col-major [2I, S] view)
  void* dact_ = nullptr;     // bf16 [mpc, I]
  float* dgemv_ = nullptr;   // S=1 GEMV partials (max of the two sites' needs)
  rocblas_handle handle_ = nullptr;  // borrowed from the Model's Resources
};

}  // namespace alice::gpu
