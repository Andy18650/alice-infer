#pragma once

// GPU platform plumbing: lazy singleton holding the device + rocBLAS handle
// for the GPU-resident path (single-GPU migration step: device 0).

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

}  // namespace alice::gpu_rocblas
