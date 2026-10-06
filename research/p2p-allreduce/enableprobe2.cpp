// Stage 5: engine-faithful: threads + hipMalloc + LARGE PAGEABLE H2D copy from
// a THP mmap (the loader's staging pattern) -> then serial peer enable.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <sys/mman.h>
#include <hip/hip_runtime.h>
int main() {
  std::atomic<int> turn{0};
  auto worker = [&](int r) {
    hipSetDevice(r);
    float* a; hipMalloc(&a, 256 << 20);
    // THP staging + pageable H2D (exactly the loader's pattern)
    size_t sz = 64 << 20;
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(p, sz, MADV_HUGEPAGE);
    memset(p, 1, sz);  // first-touch
    hipMemcpy(a, p, sz, hipMemcpyHostToDevice);   // pageable H2D
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
