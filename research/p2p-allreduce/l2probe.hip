// Does a peer's remote WRITE to my VRAM invalidate my L2? (Your register design.)
// GCD0 buf0=A, GCD1 buf1=B. Each reads the PEER's buf (saves to a buffer),
// then writes to the PEER's buf, sync, then reads OWN buf locally and compares
// against the saved peer value. If local read == saved (peer's value) -> peer
// write visible (L2 invalidated/bypassed). If local read == own old value ->
// L2 stale (peer write not visible to local read).
#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
__global__ void k_test(float* my_buf, const float* peer_buf, float* peer_buf_w,
                       volatile uint32_t* my_flag, volatile const uint32_t* peer_flag,
                       float* saved, uint32_t* mism, uint32_t n, float my_init) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= n) return;
  // Phase 1: read peer's buf (peer read) -> save
  saved[i] = peer_buf[i];
  // sync: both done reading
  __threadfence_system();
  if (threadIdx.x == 0) *my_flag = 1;
  while (*peer_flag < 1) {}
  // Phase 2: write my_init to peer's buf (peer write)
  peer_buf_w[i] = my_init;
  // sync: both done writing (system fence)
  __threadfence_system();
  if (threadIdx.x == 0) atomicExch((uint32_t*)my_flag, 2u);
  while (*peer_flag < 2) {}
  // Phase 3: read my OWN buf (local) -> compare against saved (peer's init)
  float local = my_buf[i];
  if (local != saved[i]) atomicAdd(mism, 1u);
}
int main() {
  uint32_t n = 8192;
  float* buf0; hipSetDevice(0); hipMalloc(&buf0, n*4);
  float* buf1; hipSetDevice(1); hipMalloc(&buf1, n*4);
  // init: buf0=100, buf1=200
  hipSetDevice(0); hipMemsetAsync(buf0, 100, n*4); 
  hipSetDevice(1); hipMemsetAsync(buf1, 200, n*4);
  // enable peer access both ways
  int can; hipSetDevice(0); hipDeviceCanAccessPeer(&can,0,1); if(can) hipDeviceEnablePeerAccess(1,0);
  hipSetDevice(1); hipDeviceCanAccessPeer(&can,1,0); if(can) hipDeviceEnablePeerAccess(0,0);
  // host-pinned flags (coherent)
  uint32_t *f0, *f1;
  hipHostMalloc((void**)&f0, 4); hipHostMalloc((void**)&f1, 4);
  *f0 = 0; *f1 = 0;
  // saved + mismatch per GPU
  float *sav0, *sav1; uint32_t *m0, *m1;
  hipSetDevice(0); hipMalloc(&sav0, n*4); hipMalloc(&m0, 4);
  hipSetDevice(1); hipMalloc(&sav1, n*4); hipMalloc(&m1, 4);
  hipMemsetAsync(m0, 0, 4); hipSetDevice(1); hipMemsetAsync(m1, 0, 4);
  hipDeviceSynchronize(); hipSetDevice(1); hipDeviceSynchronize();
  // launch both concurrently
  hipSetDevice(0); k_test<<<(n+255)/256,256>>>(buf0, buf1, buf1, f0, f1, sav0, m0, n, 100.0f);
  hipSetDevice(1); k_test<<<(n+255)/256,256>>>(buf1, buf0, buf0, f1, f0, sav1, m1, n, 200.0f);
  hipSetDevice(0); hipDeviceSynchronize(); hipSetDevice(1); hipDeviceSynchronize();
  uint32_t mm0=0, mm1=0;
  hipMemcpy(&mm0, m0, 4, hipMemcpyDeviceToHost); hipSetDevice(1);
  hipMemcpy(&mm1, m1, 4, hipMemcpyDeviceToHost);
  printf("GCD0: saved(peer=200) vs local(buf0 after GCD1 wrote 200): %u mismatches -> %s\n",
         mm0, mm0==0?"peer write VISIBLE (L2 invalidated/bypassed)":"STALE (local L2 not invalidated)");
  printf("GCD1: saved(peer=100) vs local(buf1 after GCD0 wrote 100): %u mismatches -> %s\n",
         mm1, mm1==0?"peer write VISIBLE (L2 invalidated/bypassed)":"STALE (local L2 not invalidated)");
  // also print actual values for sanity
  float v0, v1; hipSetDevice(0); hipMemcpy(&v0, buf0, 4, hipMemcpyDeviceToHost);
  hipSetDevice(1); hipMemcpy(&v1, buf1, 4, hipMemcpyDeviceToHost);
  printf("buf0[0]=%.0f (want 200, GCD1 wrote)  buf1[0]=%.0f (want 100, GCD0 wrote)\n", v0, v1);
  return 0;
}
