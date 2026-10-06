// Stage 6: worker threads + hipExtMallocWithFlags(fine-grained) + serial enable
#include <cstdio>
#include <atomic>
#include <thread>
#include <hip/hip_runtime.h>
int main() {
  std::atomic<int> turn{0};
  auto worker = [&](int r) {
    hipSetDevice(r);
    float* a; hipMalloc(&a, 1 << 20);
    void* fg; hipExtMallocWithFlags(&fg, 8, hipDeviceMallocFinegrained);
    hipDeviceSynchronize();
    while (turn.load() != r) {}
    for (int j = 0; j < 2; ++j) {
      if (j == r) continue;
      int can = 0; hipDeviceCanAccessPeer(&can, r, j);
      if (can) {
        hipError_t e = hipDeviceEnablePeerAccess(j, 0);
        printf("rank %d: enablePeer(%d) -> %s\n", r, j, hipGetErrorString(e)); fflush(stdout);
      }
    }
    turn.store(r + 1);
  };
  std::thread t0(worker, 0), t1(worker, 1);
  t0.join(); t1.join();
  printf("done\n");
  return 0;
}
