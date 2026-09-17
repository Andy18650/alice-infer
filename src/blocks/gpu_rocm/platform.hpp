#pragma once

// GPU platform plumbing for the ROCm path: the per-GPU resources (device
// binding + rocBLAS handle) as a plain RAII value OWNED BY THE MODEL — the
// same ownership pattern as the shared prefill scratch: session-scoped GPU
// resources live under Model, never in process-wide singletons (user
// direction 2026-09, pre-paving the multi-die worker-thread plan where each
// Model's device parameters are thread-private).
//
// hipSetDevice is PER-THREAD state: the thread that acquires the Context is
// the thread that must run the model's loads and forwards. Nothing else in
// the engine changes the current device.

#include <string>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace alice::gpu_rocm {

struct Context {
  int device = 0;
  std::string arch;  // e.g. "gfx90a" (reporting only)
  rocblas_handle handle = nullptr;

  Context() = default;
  // Validates the device exists, binds it as the calling thread's current
  // device, creates the rocBLAS handle. Throws std::runtime_error loudly.
  explicit Context(int device);
  ~Context();  // destroys the handle

  Context(Context&& o) noexcept;
  Context& operator=(Context&& o) noexcept;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

}  // namespace alice::gpu_rocm
