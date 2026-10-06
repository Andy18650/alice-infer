// Where do PEER-read lines allocate on the reader: L1 only, or also L2?
// GCD1: 16 co-resident blocks. Block 0 warm-reads the buffer (peer or local),
// flags the others, then re-reads timed (warm L1). Blocks 1..15 (cold L1, but
// the shared L2 slice would be warm IF peer reads allocate in L2) read timed.
// If blocks 1..15 ~ block0-warm speed -> L2 allocation. If ~fabric latency ->
// L1-only allocation (L2 does not cache fabric-read data).
#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
__global__ void k_alloc(const float* buf, unsigned int* flag,
                        unsigned long long* times, unsigned int n) {
  if (threadIdx.x != 0) return;
  volatile const float* vb = (volatile const float*)buf;
  if (blockIdx.x == 0) {
    float s = 0;
    for (unsigned i = 0; i < n; ++i) s += vb[i];          // warm read (untimed)
    __threadfence();
    atomicExch(flag, 1u);
    unsigned long long t0 = wall_clock64();                // timed re-read (warm L1)
    float s2 = 0;
    for (unsigned i = 0; i < n; ++i) s2 += vb[i];
    unsigned long long t1 = wall_clock64();
    times[0] = t1 - t0;
    if (s + s2 == 12345.f) times[31] = 1;
  } else {
    while (atomicAdd(flag, 0u) < 1) {}
    unsigned long long t0 = wall_clock64();                // cold L1; L2 ???
    float s = 0;
    for (unsigned i = 0; i < n; ++i) s += vb[i];
    unsigned long long t1 = wall_clock64();
    times[blockIdx.x] = t1 - t0;
    if (s == 12345.f) times[30] = 1;
  }
}
void run(const char* label, int reader_dev, int owner_dev) {
  unsigned int n = 2048;  // 8 KB: fits in one CU's 16 KB L1
  float* buf; hipSetDevice(owner_dev); hipMalloc(&buf, n*4); hipMemsetAsync(buf, 42, n*4);
  unsigned int* flag; unsigned long long* times;
  hipSetDevice(reader_dev); hipMalloc(&flag, 4); hipMalloc(&times, 32*8);
  hipMemsetAsync(flag, 0, 4); hipMemsetAsync(times, 0, 32*8);
  if (reader_dev != owner_dev) {
    int can; hipDeviceCanAccessPeer(&can, reader_dev, owner_dev);
    if (can) hipDeviceEnablePeerAccess(owner_dev, 0);
  }
  hipSetDevice(owner_dev); hipDeviceSynchronize(); hipSetDevice(reader_dev); hipDeviceSynchronize();
  hipSetDevice(reader_dev);
  k_alloc<<<16, 32>>>(buf, flag, times, n);
  hipDeviceSynchronize();
  unsigned long long h[32];
  hipMemcpy(h, times, 32*8, hipMemcpyDeviceToHost);
  printf("%s: block0(warm L1) %.0f ns/load | cold-L1 blocks:", label, h[0]*40.0/n);
  double mn = 1e18, mx = 0;
  for (int b = 1; b < 16; ++b) { double v = h[b]*40.0/n; if(v<mn)mn=v; if(v>mx)mx=v; }
  printf(" min %.0f  max %.0f ns/load\n", mn, mx);
  hipSetDevice(owner_dev); hipFree(buf);
  hipSetDevice(reader_dev); hipFree(flag); hipFree(times);
}
int main() {
  run("LOCAL read (control) ", 1, 1);
  run("PEER read (GCD1->GCD0)", 1, 0);
  return 0;
}
