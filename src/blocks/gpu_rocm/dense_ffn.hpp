#pragma once

// GPU-resident dense FFN block (design §8.1; gfx90a implementation:
// rocBLAS gemm_ex bf16→fp32 + SIMT silu-mul). Device-io successor of the
// Device-resident dense FFN: same math as the cpu_generic oracle (fused
// silu-mul, down GEMM into io), no host staging — the stream never leaves
// the device.

#include "../../gpu/block.hpp"

#include "alice/engine_config.hpp"
#include "alice/manifest.hpp"
#include "alice/tensor.hpp"

namespace alice::gpu_rocm {

class DenseFfnBF16 final : public gpu::Block {
 public:
  // Weight views are validated staging buffers, considered freed on return
  // (fused/copied to the device in the constructor).
  DenseFfnBF16(BF16View w_gate, BF16View w_up, BF16View w_down, const ModelConfig& mc,
               const EngineConfig& ec);
  ~DenseFfnBF16() override;

  void forward(float* io, uint32_t S) override;

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;
  void* dw_gup_ = nullptr;   // bf16 [2I, d] (fused [Wg;Wu])
  void* dw_down_ = nullptr;  // bf16 [d, I]
  void* dx_ = nullptr;       // bf16 [mpc, d]
  float* dgu_ = nullptr;     // fp32 [mpc, 2I] (col-major [2I, S] view)
  void* dact_ = nullptr;     // bf16 [mpc, I]
};

}  // namespace alice::gpu
