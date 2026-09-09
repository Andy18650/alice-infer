#pragma once

// CPU_AVX2 implementation of DenseFfnBF16 — the first self-contained
// optimized operator (design §10): fully private weight storage and math,
// no shared toolbox code. The inner kernels (dot/axpy) live in the .cpp as
// isolated functions so step 3 can swap in AVX2+FMA intrinsics without
// another structural rewrite.

#include <cstdlib>
#include <memory>

#include "alice/dense_ffn_bf16.hpp"

namespace alice::cpu_avx2 {

// Aligned (64B) private storage: AVX2 loads/stores and page-granularity
// first-touch both want this.
struct AlignedDeleter {
  void operator()(void* p) const { std::free(p); }
};
template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter>;

class DenseFfnBF16CpuAvx2 final : public DenseFfnBF16 {
 public:
  // Abstract-class constructor contract: views validated, freed on return.
  DenseFfnBF16CpuAvx2(BF16View w_gate, BF16View w_up, BF16View w_down,
                      const ModelConfig& mc, const EngineConfig& ec);

  void forward(F32View io) override;

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;
  size_t ldw_ = 0;  // gate/up/transposed-down row stride (padded to 64B)
  size_t ldi_ = 0;  // row-major w_down row stride (padded to 64B)

  AlignedPtr<bf16> wg_, wu_;   // [I, ldw], copied with parallel first-touch
  AlignedPtr<bf16> wdt_;       // w_down TRANSPOSED: [I, ldw] (Wd[i][j] -> WdT[j][i])
  AlignedPtr<bf16> wd_;        // w_down row-major [d, ldi] (S==1 dot path)
  AlignedPtr<float> act_;      // [chunk, mpc] fused-chunk activations
  AlignedPtr<float> full_act_; // [I] full activation vector (S==1 path)
  AlignedPtr<float> out_;      // [mpc, d] private output accumulator
};

}  // namespace alice::cpu_avx2
