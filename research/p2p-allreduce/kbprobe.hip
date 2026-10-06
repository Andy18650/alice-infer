// "Coarse peer buffers fresh at kernel boundaries"? Test whether a KERNEL
// BOUNDARY alone (no system-fence event) invalidates the reader's L2.
// GCD1 primes its L2 reading buf0 (peer, =100). GCD0 writes buf0=200 (local).
// GCD1 reads buf0 (peer) in a NEW kernel. Order via NO-FENCE events
// (hipEventDisableSystemFence) so ONLY the kernel boundary could invalidate.
// read==200 -> kernel boundary invalidates (claim holds). read==100 -> stale
// (the event fence, not the kernel boundary, is what invalidates).
#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
__global__ void k_prime(const float* buf, float* sink, uint32_t n) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < n) sink[i] = buf[i];          // peer read -> caches in reader's L2
}
__global__ void k_write(float* buf, float v, uint32_t n) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < n) buf[i] = v;                // local write
}
__global__ void k_check(const float* buf, float* out, uint32_t* mism, uint32_t n) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < n) { float v = buf[i]; out[i] = v; if (v != 200.0f) atomicAdd(mism, 1u); }
}
int run(unsigned evflags, const char* label) {
  uint32_t n = 8192;
  float* buf0; hipSetDevice(0); hipMalloc(&buf0, n*4);  // GCD0's buffer
  hipMemsetAsync(buf0, 100, n*4);                        // buf0 = 100
  float *sink1, *out1; uint32_t* m1;
  hipSetDevice(1); hipMalloc(&sink1, n*4); hipMalloc(&out1, n*4); hipMalloc(&m1, 4);
  hipMemsetAsync(m1, 0, 4);
  int can; hipSetDevice(1); hipDeviceCanAccessPeer(&can,1,0); if(can) hipDeviceEnablePeerAccess(0,0);
  hipSetDevice(0); hipDeviceSynchronize(); hipSetDevice(1); hipDeviceSynchronize();
  hipEvent_t evP, evW;
  hipSetDevice(1); hipEventCreateWithFlags(&evP, evflags);
  hipSetDevice(0); hipEventCreateWithFlags(&evW, evflags);
  uint32_t blocks = (n+255)/256;
  // GCD1 prime: peer-read buf0 (caches 100 in GCD1's L2)
  hipSetDevice(1); k_prime<<<blocks,256>>>(buf0, sink1, n); hipEventRecord(evP);
  // GCD0 write: local write buf0=200
  hipSetDevice(0); hipStreamWaitEvent(0, evP, 0); k_write<<<blocks,256>>>(buf0, 200.0f, n); hipEventRecord(evW);
  // GCD1 check: peer-read buf0 in a NEW kernel
  hipSetDevice(1); hipStreamWaitEvent(0, evW, 0); k_check<<<blocks,256>>>(buf0, out1, m1, n);
  hipDeviceSynchronize(); hipSetDevice(0); hipDeviceSynchronize();
  uint32_t mm=0; hipSetDevice(1); hipMemcpy(&mm, m1, 4, hipMemcpyDeviceToHost);
  float v0; hipMemcpy(&v0, out1, 4, hipMemcpyDeviceToHost);
  printf("%-28s read[0]=%.0f  mismatches=%u -> %s\n", label, v0, mm,
         mm==0 ? "FRESH (kernel boundary invalidated L2)" : "STALE (L2 persists across kernels)");
  hipSetDevice(0); hipFree(buf0); hipSetDevice(1); hipFree(sink1); hipFree(out1); hipFree(m1);
  return mm;
}
int main() {
  run(hipEventDisableSystemFence|hipEventDisableTiming, "no-fence events");
  run(hipEventDisableTiming, "default (system-fence) events");
  return 0;
}
