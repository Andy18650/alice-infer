#include "alice/dense_ffn_bf16.hpp"

#include <stdexcept>

#include "cpu_avx2/dense_ffn.hpp"
#include "cpu_generic/dense_ffn.hpp"
#if defined(ALICE_HAVE_ROCBLAS)
#include "gpu_rocblas/dense_ffn.hpp"
#endif

namespace alice {

std::unique_ptr<DenseFfnBF16> DenseFfnBF16::create(HardwareClass hw, BF16View w_gate,
                                                   BF16View w_up, BF16View w_down,
                                                   const ModelConfig& mc,
                                                   const EngineConfig& ec) {
  switch (hw) {
    case HardwareClass::AUTO:  // auto-select: GPU when available, then CPU chain
    case HardwareClass::GPU_ROCBLAS:
#if defined(ALICE_HAVE_ROCBLAS)
      if (gpu_available())
        return std::make_unique<gpu_rocblas::DenseFfnBF16GpuRocblas>(w_gate, w_up,
                                                                     w_down, mc, ec);
#endif
      [[fallthrough]];  // no ROCm build or no GPU: resolve down the CPU chain
    case HardwareClass::CPU_AVX2:
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__)
      // Runtime guard: the AVX2 TU is compiled with -mavx2 -mfma; only
      // dispatch when the machine actually has both. Otherwise fall through
      // to the oracle (the factory fallback chain, design §7.1).
      if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return std::make_unique<cpu_avx2::DenseFfnBF16CpuAvx2>(w_gate, w_up, w_down, mc,
                                                               ec);
#endif
      [[fallthrough]];
    case HardwareClass::CPU_GENERIC:
      return std::make_unique<cpu_generic::DenseFfnBF16CpuGeneric>(w_gate, w_up, w_down, mc,
                                                                   ec);
    default:
      throw std::invalid_argument(
          "DenseFfnBF16::create: hardware class not implemented (fallback chain ends at "
          "CPU_GENERIC)");
  }
}

}  // namespace alice
