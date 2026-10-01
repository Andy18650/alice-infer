// gpu::Resources impl (PROTOTYPE 2026-09 — NOT frozen). See resources.hpp.

#include "resources.hpp"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace alice::gpu {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("gpu::Resources: " + what);
}

#define HIP_CHECK(x)                                                            \
  do {                                                                         \
    const hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess)                                                      \
      fail(std::string(#x) + " failed: " + hipGetErrorString(e_));             \
  } while (0)

size_t total_free_device_bytes() {
  size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) != hipSuccess) return 0;
  return free;
}

}  // namespace

Resources::Resources(int dev) : device_(dev) {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  if (dev < 0 || dev >= n)
    fail("no ROCm device " + std::to_string(dev) + " (found " +
         std::to_string(n) + ")");
  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  arch_ = prop.gcnArchName;  // "gfx90a:sramecc+:xnack-" -> "gfx90a"
  arch_ = arch_.substr(0, arch_.find(':'));
  if (rocblas_create_handle(&handle_) != rocblas_status_success)
    fail("rocblas_create_handle failed");
}

Resources::~Resources() {
  for (auto& [p, n] : ledger_)
    if (p) (void)hipFree(p);
  for (auto& [name, np] : named_)
    if (np.first) (void)hipFree(np.first);
  if (scratch_) (void)hipFree(scratch_);
  if (handle_) (void)rocblas_destroy_handle(handle_);
}

void* Resources::alloc(size_t bytes) {
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) {
    const size_t free = total_free_device_bytes();
    fail("hipMalloc(" + std::to_string(bytes) + ") failed; " +
         std::to_string(free >> 20) + " MiB free on device " +
         std::to_string(device_) + ", ledger holds " +
         std::to_string(ledger_.size()) + " allocations");
  }
  ledger_.push_back({p, bytes});
  return p;
}

void Resources::init_scratch(size_t bytes) {
  if (scratch_) fail("scratch already initialized");
  scratch_bytes_ = bytes;
  if (bytes == 0) return;
  if (hipMalloc(reinterpret_cast<void**>(&scratch_), bytes) != hipSuccess)
    fail("scratch hipMalloc(" + std::to_string(bytes) + ") failed");
  // Poison fill (quiet NaN): read-before-first-write becomes visible.
  if (hipMemsetD32(scratch_, 0x7fc00000, uint32_t(bytes / 4)) != hipSuccess)
    fail("scratch poison fill failed");
}

void* Resources::named(const std::string& name, size_t bytes) {
  for (auto& [n, np] : named_) {
    if (n == name) {
      if (np.second != bytes)
        fail("named buffer '" + name + "' requested at " +
             std::to_string(bytes) + " bytes but exists at " +
             std::to_string(np.second) + " bytes");
      return np.first;
    }
  }
  void* p = alloc(bytes);  // reuses alloc's ledger + OOM context
  named_.push_back({name, {p, bytes}});
  return p;
}

}  // namespace alice::gpu
