#include <cstdio>
#include <cstdint>
#include <chrono>
#include <hip/hip_runtime.h>
// volatile forces a real load every rep (no CSE). If peer reads are L2-cached,
// bandwidth ~= local L2 (high). If uncached, ~= fabric (~134 GB/s).
__global__ void k_bw(volatile const float* buf, float* sink, uint32_t n, uint32_t reps) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  float s = 0;
  for (uint32_t r = 0; r < reps; ++r)
    if (i < n) s += buf[i];   // volatile: real load every rep, same address
  if (i < n) sink[i] = s;
}
double bench(int dev, float* buf, uint32_t n, uint32_t reps) {
  hipSetDevice(dev);
  float* sink; hipMalloc(&sink, n*4);
  k_bw<<<(n+255)/256,256>>>(buf, sink, n, 1); hipDeviceSynchronize();  // warm
  auto t0 = std::chrono::steady_clock::now();
  k_bw<<<(n+255)/256,256>>>(buf, sink, n, reps); hipDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  double s = std::chrono::duration<double>(t1-t0).count();
  hipFree(sink);
  return (double)n * reps * 4 / s / 1e9;
}
int main() {
  uint32_t n = 8192;  // 32 KB, fits in L2
  uint32_t reps = 100000;
  float* buf0; hipSetDevice(0); hipMalloc(&buf0, n*4); hipMemsetAsync(buf0, 42, n*4);
  float* buf1; hipSetDevice(1); hipMalloc(&buf1, n*4); hipMemsetAsync(buf1, 99, n*4);
  int can; hipSetDevice(0); hipDeviceCanAccessPeer(&can,0,1); if(can) hipDeviceEnablePeerAccess(1,0);
  hipSetDevice(1); hipDeviceCanAccessPeer(&can,1,0); if(can) hipDeviceEnablePeerAccess(0,0);
  hipSetDevice(0); hipDeviceSynchronize(); hipSetDevice(1); hipDeviceSynchronize();
  printf("buffer=%u floats (%u KB), %u reps (volatile loads)\n", n, n*4/1024, reps);
  printf("local read  (GCD0 -> buf0):   %8.1f GB/s\n", bench(0, buf0, n, reps));
  printf("peer read   (GCD0 -> buf1@GCD1): %8.1f GB/s\n", bench(0, buf1, n, reps));
  printf("peer read   (GCD1 -> buf0@GCD0): %8.1f GB/s\n", bench(1, buf0, n, reps));
  printf("(fabric ceiling ~134 GB/s; L2 would be >> 134)\n");
  return 0;
}
