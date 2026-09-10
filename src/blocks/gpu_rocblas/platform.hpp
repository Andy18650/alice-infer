#pragma once

// GPU platform plumbing for the GPU_ROCBLAS hardware class (first GPU =
// device 0, interim until the loader overhaul): lazy singleton holding the
// rocBLAS handle + device arch, plus hipRTC kernel compilation (keeps hipcc
// out of the build; kernels ship as source strings inside each impl —
// platform plumbing lives here, impl math stays in the impls).

#include <string>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace alice::gpu_rocblas {

bool available_impl();  // device 0 exists; cached

struct Platform {
  int device = 0;
  std::string arch;  // e.g. "gfx90a" (feature suffixes stripped)
  rocblas_handle handle = nullptr;
};

Platform& platform();  // initializes on first use; throws std::runtime_error

// Compile `src` (one extern "C" __global__ function named `name`) for the
// platform arch via hipRTC and load it; cached by name. Throws
// std::runtime_error with the compiler log on failure.
hipFunction_t rtc_kernel(const char* name, const char* src);

}  // namespace alice::gpu_rocblas
