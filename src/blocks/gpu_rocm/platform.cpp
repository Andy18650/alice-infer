#include "platform.hpp"

#include <stdexcept>
#include <utility>

namespace alice::gpu_rocm {

namespace {
#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    const hipError_t e_ = (x);                                                           \
    if (e_ != hipSuccess)                                                                \
      throw std::runtime_error(std::string("gpu_rocm: HIP error: ") +                    \
                               hipGetErrorString(e_) + " at " __FILE__ ":" +             \
                               std::to_string(__LINE__));                                \
  } while (0)
}  // namespace

Context::Context(int dev) : device(dev) {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  if (dev < 0 || dev >= n)
    throw std::runtime_error("gpu_rocm: no ROCm device " + std::to_string(dev) +
                             " (found " + std::to_string(n) + ")");
  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  arch = prop.gcnArchName;  // "gfx90a:sramecc+:xnack-" -> "gfx90a"
  arch = arch.substr(0, arch.find(':'));
  if (rocblas_create_handle(&handle) != rocblas_status_success)
    throw std::runtime_error("gpu_rocm: rocblas_create_handle failed");
}

Context::~Context() {
  if (handle) (void)rocblas_destroy_handle(handle);
}

Context::Context(Context&& o) noexcept
    : device(o.device), arch(std::move(o.arch)), handle(o.handle) {
  o.handle = nullptr;
}

Context& Context::operator=(Context&& o) noexcept {
  if (this != &o) {
    if (handle) (void)rocblas_destroy_handle(handle);
    device = o.device;
    arch = std::move(o.arch);
    handle = o.handle;
    o.handle = nullptr;
  }
  return *this;
}

}  // namespace alice::gpu_rocm
