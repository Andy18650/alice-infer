#include "platform.hpp"

#include <stdexcept>

namespace alice::gpu_rocm {

namespace {
#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    const hipError_t e_ = (x);                                                           \
    if (e_ != hipSuccess)                                                                \
      throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e_) +      \
                               " at " __FILE__ ":" + std::to_string(__LINE__));          \
  } while (0)
}  // namespace

bool available_impl() {
  static const bool ok = [] {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
  }();
  return ok;
}

Platform& platform() {
  static Platform p = [] {
    if (!available_impl()) throw std::runtime_error("gpu_rocm: no ROCm GPU present");
    Platform q;
    HIP_CHECK(hipSetDevice(q.device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, q.device));
    q.arch = prop.gcnArchName;  // "gfx90a:sramecc+:xnack-" -> "gfx90a"
    q.arch = q.arch.substr(0, q.arch.find(':'));
    if (rocblas_create_handle(&q.handle) != rocblas_status_success)
      throw std::runtime_error("gpu_rocm: rocblas_create_handle failed");
    return q;
  }();
  return p;
}

}  // namespace alice::gpu_rocm
