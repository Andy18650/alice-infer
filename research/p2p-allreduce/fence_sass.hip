#include <hip/hip_runtime.h>
__global__ void k(float* data, unsigned int* flag, unsigned int v) {
  data[threadIdx.x] = v;                       // payload write (normal store)
  __threadfence_system();                      // what does THIS emit?
  if (threadIdx.x == 0)
    __hip_atomic_store(flag, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
