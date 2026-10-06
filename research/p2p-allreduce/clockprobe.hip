#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
#define HIPCK(x) do{hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); return 1;}}while(0)
// wall_clock64() = system-scope steady counter (40 ns/tick on MI250X, 25 MHz);
// hipDeviceAttributeWallClockRate is in kHz. Coherent across GCDs (same rate,
// common timebase) — so (T_reader - T_writer)/rate is a true cross-GCD latency.
__global__ void k_writer(volatile uint64_t* flag, uint64_t* tout){
  if(threadIdx.x==0){ uint64_t t=wall_clock64(); __threadfence_system(); *flag=1; *tout=t; }
}
__global__ void k_reader(volatile const uint64_t* flag, uint64_t* tout){
  if(threadIdx.x==0){ uint64_t t; do{ t=wall_clock64(); }while(*flag==0); *tout=t; }
}
static double probe(int wdev, int rdev){
  int rw=0, rr=0;
  hipDeviceGetAttribute(&rw, hipDeviceAttributeWallClockRate, wdev);
  hipDeviceGetAttribute(&rr, hipDeviceAttributeWallClockRate, rdev);
  uint64_t* hflag; hipHostMalloc((void**)&hflag, sizeof(uint64_t));
  uint64_t* tw; hipSetDevice(wdev); hipMalloc(&tw, sizeof(uint64_t));
  uint64_t* tr; hipSetDevice(rdev); hipMalloc(&tr, sizeof(uint64_t));
  *hflag=0; hipSetDevice(wdev); hipDeviceSynchronize(); hipSetDevice(rdev); hipDeviceSynchronize();
  hipSetDevice(rdev); k_reader<<<1,1>>>(hflag, tr);   // reader spins first
  hipSetDevice(wdev); k_writer<<<1,1>>>(hflag, tw);    // writer sets flag + T
  hipSetDevice(wdev); hipDeviceSynchronize(); hipSetDevice(rdev); hipDeviceSynchronize();
  uint64_t a=0,b=0; hipMemcpy(&a, tw, 8, hipMemcpyDeviceToHost);
  hipSetDevice(rdev); hipMemcpy(&b, tr, 8, hipMemcpyDeviceToHost);
  // rate in kHz -> us = ticks * 1000 / rate_kHz
  double us = (double)(int64_t)(b - a) * 1000.0 / (double)rw;
  printf("GCD%d->GCD%d  rate=%d/%d kHz  delta=%lld ticks = %.3f us\n",
         wdev, rdev, rw, rr, (long long)(b-a), us);
  hipFree(tw); hipSetDevice(rdev); hipFree(tr); hipHostFree(hflag);
  return us;
}
int main(){
  printf("== intra-card (GCD0<->GCD1, same MI250X) ==\n");
  probe(0,1); probe(1,0);
  printf("== cross-card (GCD0<->GCD2, different MI250X, PCIe) ==\n");
  probe(0,2); probe(2,0);
  return 0;
}
