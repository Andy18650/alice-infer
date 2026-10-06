// Does __hip_atomic_load(ACQUIRE/RELAXED, SYSTEM) see through a stale L2?
// Same scenario as l2probe: GCD0 caches buf0=100 in its L2 (prime read),
// GCD1 peer-writes buf0=200, then GCD0 reads buf0 locally with:
//   A) normal load           (control, expect stale 100)
//   B) atomic relaxed system (glc)
//   C) atomic acquire system (glc + fence)
#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
__device__ void flag_set(volatile unsigned int* f, unsigned int v) {
  __hip_atomic_store((unsigned int*)f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void flag_wait(volatile unsigned int* f, unsigned int v) {
  while (__hip_atomic_load((unsigned int*)f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v) {}
}
template <int MODE>
__global__ void k_test(float* my_buf, float* peer_buf_w,
                       volatile unsigned int* my_flag, volatile unsigned int* peer_flag,
                       float* out, uint32_t* mism, uint32_t n) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= n) return;
  float prime = my_buf[i];          // cache my_buf (=100) in MY L2
  if (threadIdx.x == 0) flag_set(my_flag, 1);
  flag_wait(peer_flag, 1);          // both primed
  peer_buf_w[i] = 200.0f;           // peer write 200 into the other's buffer
  __threadfence_system();
  if (threadIdx.x == 0) flag_set(my_flag, 2);
  flag_wait(peer_flag, 2);          // both wrote
  float v;
  if (MODE == 0) v = my_buf[i];     // normal load
  else if (MODE == 1) v = __hip_atomic_load((unsigned int*)my_buf + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM), v = __uint_as_float(__float_as_uint(v));
  else v = __uint_as_float(__hip_atomic_load((unsigned int*)my_buf + i, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM));
  if (MODE == 1) v = __uint_as_float(__hip_atomic_load((unsigned int*)my_buf + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
  out[i] = v + prime*0.0f;          // keep prime alive
  if (v != 200.0f) atomicAdd(mism, 1u);
}
template <int MODE>
int run(const char* label) {
  uint32_t n = 8192;
  float *b0, *b1, *o0, *o1; uint32_t *m0, *m1;
  hipSetDevice(0); hipMalloc(&b0, n*4); hipMemsetAsync(b0, 100, n*4); hipMalloc(&o0, n*4); hipMalloc(&m0, 4); hipMemsetAsync(m0, 0, 4);
  hipSetDevice(1); hipMalloc(&b1, n*4); hipMemsetAsync(b1, 100, n*4); hipMalloc(&o1, n*4); hipMalloc(&m1, 4); hipMemsetAsync(m1, 0, 4);
  int can; hipSetDevice(0); hipDeviceCanAccessPeer(&can,0,1); if(can) hipDeviceEnablePeerAccess(1,0);
  hipSetDevice(1); hipDeviceCanAccessPeer(&can,1,0); if(can) hipDeviceEnablePeerAccess(0,0);
  unsigned int *f0, *f1;
  hipHostMalloc((void**)&f0, 4); hipHostMalloc((void**)&f1, 4); *f0 = 0; *f1 = 0;
  hipSetDevice(0); hipDeviceSynchronize(); hipSetDevice(1); hipDeviceSynchronize();
  hipSetDevice(0); k_test<MODE><<<(n+255)/256,256>>>(b0, b1, f0, f1, o0, m0, n);
  hipSetDevice(1); k_test<MODE><<<(n+255)/256,256>>>(b1, b0, f1, f0, o1, m1, n);
  hipSetDevice(0); hipDeviceSynchronize(); hipSetDevice(1); hipDeviceSynchronize();
  uint32_t mm0, mm1; hipSetDevice(0); hipMemcpy(&mm0, m0, 4, hipMemcpyDeviceToHost);
  hipSetDevice(1); hipMemcpy(&mm1, m1, 4, hipMemcpyDeviceToHost);
  printf("%-40s GCD0 mism=%u  GCD1 mism=%u -> %s\n", label, mm0, mm1,
         (mm0==0&&mm1==0) ? "FRESH" : "STALE");
  return mm0+mm1;
}
int main() {
  run<0>("A: normal local read (control)");
  run<1>("B: atomic load RELAXED SYSTEM");
  run<2>("C: atomic load ACQUIRE SYSTEM");
  return 0;
}
