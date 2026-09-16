#include "trunk.hpp"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

#include "../blocks/gpu_rocm/platform.hpp"
#include "kernels.hpp"

namespace alice::gpu {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::RmsNormAddTrunk: ") + #x +            \
                               " failed: " + hipGetErrorString(e_));                    \
  } while (0)

void h2d_f32_from_bf16(BF16View src, float* dst_dev, uint32_t d, const char* what) {
  if (src.numel() != d)
    throw std::invalid_argument(std::string("gpu::RmsNormAddTrunk: bad norm: ") + what);
  std::vector<float> host(d);
  for (uint32_t i = 0; i < d; ++i) host[i] = src.data[i].to_f32();
  HIP_CHECK(hipMemcpy(dst_dev, host.data(), d * sizeof(float), hipMemcpyHostToDevice));
}
}  // namespace

RmsNormAddTrunk::RmsNormAddTrunk(std::vector<BF16View> pre_norms, BF16View final_norm,
                                 float eps, uint32_t hidden, uint32_t max_prefill_chunk)
    : d_(hidden), mpc_(max_prefill_chunk), eps_(eps) {
  gpu_rocm::Platform& p = gpu_rocm::platform();
  HIP_CHECK(hipSetDevice(p.device));
  HIP_CHECK(hipMalloc(&residual_, size_t(mpc_) * d_ * sizeof(float)));
  HIP_CHECK(hipMalloc(&io_, size_t(mpc_) * d_ * sizeof(float)));
  HIP_CHECK(hipMalloc(&norms_, pre_norms.size() * d_ * sizeof(float)));
  HIP_CHECK(hipMalloc(&final_, d_ * sizeof(float)));
  for (size_t i = 0; i < pre_norms.size(); ++i)
    h2d_f32_from_bf16(pre_norms[i], norms_ + i * d_, d_, "pre_norm");
  h2d_f32_from_bf16(final_norm, final_, d_, "final_norm");
}

RmsNormAddTrunk::~RmsNormAddTrunk() {
  (void)hipFree(residual_);
  (void)hipFree(io_);
  (void)hipFree(norms_);
  (void)hipFree(final_);
}

float* RmsNormAddTrunk::begin_pass(uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("gpu::RmsNormAddTrunk: S out of range (max_prefill_chunk)");
  S_ = S;
  return residual_;
}

void RmsNormAddTrunk::read(size_t block) {
  kernels::rmsnorm(residual_, norms_ + block * d_, io_, S_, d_, eps_);
}

void RmsNormAddTrunk::write(size_t /*block*/) {
  kernels::add_rows(residual_, io_, size_t(S_) * d_);
}

void RmsNormAddTrunk::output() { kernels::rmsnorm(residual_, final_, io_, S_, d_, eps_); }

}  // namespace alice::gpu
