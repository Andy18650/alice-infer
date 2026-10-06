// Which engine allocation pattern breaks hipDeviceEnablePeerAccess?
// stage 1: threads + hipMalloc only
// stage 2: + rocblas handle
// stage 3: + THP mmap + hipHostRegister
// stage 4: + fine-grained device alloc
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <sys/mman.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("  %s -> %s\n", #x, hipGetErrorString(e));} } while(0)
int main(int argc, char** argv) {
  int stage = argc > 1 ? atoi(argv[1]) : 1;
  auto worker = [&](int r) {
    hipSetDevice(r);
    float *a, *b;
    CK(hipMalloc(&a, 1 << 20));
    CK(hipMalloc(&b, 256 << 20));
    void* fg = nullptr;
    if (stage >= 4) CK(hipExtMallocWithFlags(&fg, 8, hipDeviceMallocFinegrained));
    rocblas_handle h = nullptr;
    if (stage >= 2) rocblas_create_handle(&h);
    void* mmap_p = nullptr;
    if (stage >= 3) {
      mmap_p = mmap(nullptr, 64 << 20, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
      if (mmap_p == MAP_FAILED) mmap_p = mmap(nullptr, 64 << 20, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      madvise(mmap_p, 64 << 20, MADV_HUGEPAGE);
      memset(mmap_p, 1, 64 << 20);
      CK(hipHostRegister(mmap_p, 64 << 20, hipHostRegisterDefault));
    }
    hipDeviceSynchronize();
    // serial enable from the owning thread: rank 0 first, then rank 1
    static std::atomic<int> turn{0};
    while (turn.load() != r) {}
    for (int j = 0; j < 2; ++j) {
      if (j == r) continue;
      int can = 0;
      hipError_t e1 = hipDeviceCanAccessPeer(&can, r, j);
      printf("rank %d: canAccess(%d)=%d err=%s\n", r, j, can, hipGetErrorString(e1));
      if (can) {
        hipError_t e2 = hipDeviceEnablePeerAccess(j, 0);
        printf("rank %d: enablePeer(%d) -> %s\n", r, j, hipGetErrorString(e2));
        fflush(stdout);
      }
    }
    turn.store(r + 1);
  };
  std::thread t0(worker, 0), t1(worker, 1);
  t0.join(); t1.join();
  printf("stage %d done\n", stage);
  return 0;
}
