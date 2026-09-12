#include "embedder.hpp"

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
      throw std::runtime_error(std::string("gpu::Embedder: ") + #x + " failed: " +     \
                               hipGetErrorString(e_));                                  \
  } while (0)
}  // namespace

Embedder::Embedder(BF16View table, uint32_t max_prefill_chunk)
    : vocab_(table.rows), d_(table.cols), mpc_(max_prefill_chunk) {
  gpu_rocm::Platform& p = gpu_rocm::platform();
  HIP_CHECK(hipSetDevice(p.device));
  HIP_CHECK(hipMalloc(&table_, table.nbytes()));
  HIP_CHECK(hipMalloc(&ids_dev_, mpc_ * sizeof(int32_t)));
  HIP_CHECK(hipMemcpy(table_, table.data, table.nbytes(), hipMemcpyHostToDevice));
}

Embedder::~Embedder() {
  (void)hipFree(table_);
  (void)hipFree(ids_dev_);
}

void Embedder::embed(std::span<const int32_t> ids, float* residual) {
  if (ids.empty() || ids.size() > mpc_)
    throw std::runtime_error("gpu::Embedder::embed: S out of range");
  for (int32_t id : ids)
    if (uint32_t(id) >= vocab_)
      throw std::out_of_range("gpu::Embedder::embed: token id out of vocab");
  HIP_CHECK(hipMemcpy(ids_dev_, ids.data(), ids.size() * sizeof(int32_t),
                      hipMemcpyHostToDevice));
  kernels::embed_gather(table_, ids_dev_, residual, uint32_t(ids.size()), d_);
}

}  // namespace alice::gpu
