#include <hip/hip_runtime.h>
// Test __hip_atomic_load / __hip_atomic_store availability + codegen on gfx90a
__device__ unsigned int ld_relaxed_sys(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ unsigned int ld_acquire_sys(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void st_release_sys(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ unsigned int ld_relaxed_agent(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__global__ void k(unsigned int* p, unsigned int* out) {
  unsigned int a = ld_relaxed_sys(p);
  unsigned int b = ld_acquire_sys(p);
  st_release_sys(p, a + b);
  out[0] = ld_relaxed_agent(p);
}
