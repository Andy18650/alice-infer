#include "cpu_reference.hpp"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

namespace alice::gpu {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::CpuReferenceBlock: ") + #x +          \
                               " failed: " + hipGetErrorString(e_));                    \
  } while (0)
}  // namespace

CpuReferenceBlock::CpuReferenceBlock(cpu_generic::DenseFfnBF16CpuGeneric b, uint32_t d,
                                     uint32_t max_prefill_chunk)
    : impl_(std::move(b)), d_(d), mpc_(max_prefill_chunk) {
  host_io_.resize(size_t(mpc_) * d_);
}

CpuReferenceBlock::CpuReferenceBlock(cpu_generic::GqaAttentionBF16CpuGeneric b, uint32_t d,
                                     uint32_t max_prefill_chunk)
    : impl_(std::move(b)), d_(d), mpc_(max_prefill_chunk) {
  host_io_.resize(size_t(mpc_) * d_);
}

void CpuReferenceBlock::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("gpu::CpuReferenceBlock::forward: S out of range");
  const size_t n = size_t(S) * d_;
  HIP_CHECK(hipMemcpy(host_io_.data(), io, n * sizeof(float), hipMemcpyDeviceToHost));
  std::visit([this, S](auto& b) { b.forward(F32View{host_io_.data(), S, d_, 0}); },
             impl_);
  HIP_CHECK(hipMemcpy(io, host_io_.data(), n * sizeof(float), hipMemcpyHostToDevice));
}

void CpuReferenceBlock::reset_state() {
  std::visit(
      [](auto& b) {
        if constexpr (requires { b.reset_state(); }) b.reset_state();
      },
      impl_);
}

}  // namespace alice::gpu
