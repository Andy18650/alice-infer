#pragma once

#include "alice/dense_ffn_bf16.hpp"

namespace alice::gpu_rocblas {

// GPU_ROCBLAS implementation of DenseFfnBF16 (first GPU = device 0, interim
// mechanism until the loader overhaul). Self-contained per design §10:
// weights staged to device memory at construction (host views freed on
// return), all math private.
//
// Structure per forward(): io (host, fp32) -> host-convert bf16 (pinned) ->
// H2D -> fused gate/up GEMM ([Wg;Wu] as one [2I, d] matrix, one launch) ->
// hipRTC silu-mul elementwise -> down GEMM -> D2H fp32 -> io rows. All
// GEMMs are rocblas_gemm_ex (bf16 in, FP32 accumulate, fp32 out); the fp32
// host io boundary means FFN-on-GPU composes freely with CPU attention.
//
// Numerics: x and the intermediate are bf16-rounded on the way in (the
// truncating conversion matches host bf16::from_f32); accumulation is fp32
// on matrix cores. Cross-impl equivalence is TOLERANCE-based (design §10);
// vs the fp32-x oracle expect bf16-epsilon-scale drift per layer.
class DenseFfnBF16GpuRocblas final : public DenseFfnBF16 {
 public:
  DenseFfnBF16GpuRocblas(BF16View w_gate, BF16View w_up, BF16View w_down,
                         const ModelConfig& mc, const EngineConfig& ec);
  ~DenseFfnBF16GpuRocblas() override;

  void forward(F32View io) override;

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;

  void* dw_gup_ = nullptr;  // device [2I, d] bf16: gate rows [0,I), up [I,2I)
  void* dw_down_ = nullptr;  // device [d, I] bf16
  void* dx_ = nullptr;      // device [d, mpc] bf16 (col-major == row-major [S, d])
  void* dgu_ = nullptr;     // device [2I, mpc] fp32 (col-major == [S, 2I])
  void* dact_ = nullptr;    // device [I, mpc] bf16
  void* dout_ = nullptr;    // device [d, mpc] fp32

  bf16* hb_x_ = nullptr;    // pinned host staging [mpc, d] bf16
  float* hb_out_ = nullptr;  // pinned host staging [mpc, d] fp32
};

}  // namespace alice::gpu_rocblas
