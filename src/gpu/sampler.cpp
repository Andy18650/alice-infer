#include "sampler.hpp"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

#include "kernels.hpp"

namespace alice::gpu {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::Sampler: ") + #x + " failed: " +      \
                               hipGetErrorString(e_));                                  \
  } while (0)
}  // namespace

Sampler::Sampler(const SamplingConfig& cfg) : cfg_(cfg), host_sampler_(cfg) {
  HIP_CHECK(hipMalloc(&id_dev_, sizeof(int32_t)));
}

Sampler::~Sampler() { (void)hipFree(id_dev_); }

int32_t Sampler::sample(const float* logits, uint32_t vocab) {
  if (cfg_.temperature == 0.0f) {
    kernels::argmax(logits, vocab, id_dev_);
    int32_t id = 0;
    HIP_CHECK(hipMemcpy(&id, id_dev_, sizeof(int32_t), hipMemcpyDeviceToHost));
    return id;
  }
  logits_host_.resize(vocab);
  HIP_CHECK(hipMemcpy(logits_host_.data(), logits, vocab * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host_sampler_.sample(logits_host_.data(), vocab);
}

}  // namespace alice::gpu
