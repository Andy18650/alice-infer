// Pure-HIP P2P viability probe (no RCCL): the fallback path for TP if RCCL
// stays broken on this stack. Verifies: (A) peer access enablement +
// hipMemcpyPeer bandwidth matrix across all 8 GCDs; (B) direct remote loads
// from a device kernel (the primitive a hand-rolled allreduce needs).
// Build: hipcc -O2 p2p.cpp -o p2p

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

#define HIP_CHECK(x)                                                     \
  do {                                                                   \
    hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess) {                                              \
      std::fprintf(stderr, "%s failed: %s (line %d)\n", #x,              \
                   hipGetErrorString(e_), __LINE__);                     \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

__global__ void remote_read(const float* __restrict__ remote,
                            float* __restrict__ local, size_t n) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t k = i; k < n; k += stride) local[k] = remote[k] * 2.0f;
}

int main() {
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  std::printf("devices=%d\n", ndev);

  // A1: capability + enablement matrix
  std::printf("can-access matrix (row can read col):\n   ");
  for (int j = 0; j < ndev; ++j) std::printf(" %d", j);
  std::printf("\n");
  for (int i = 0; i < ndev; ++i) {
    std::printf("%d: ", i);
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < ndev; ++j) {
      if (i == j) {
        std::printf(" -");
        continue;
      }
      int can = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
      std::printf(" %d", can);
      if (can) {
        const hipError_t e = hipDeviceEnablePeerAccess(j, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
          HIP_CHECK(e);
        hipGetLastError();  // clear AlreadyEnabled
      }
    }
    std::printf("\n");
  }

  const size_t bytes = 64 << 20;  // 64 MB per transfer
  const int iters = 20;
  std::vector<float*> buf(ndev, nullptr);
  for (int i = 0; i < ndev; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMalloc(&buf[i], bytes));
    HIP_CHECK(hipMemset(buf[i], 0x3f, bytes));
  }

  // A2: hipMemcpyPeer bandwidth matrix (GB/s)
  std::printf("\nhipMemcpyPeer bandwidth GB/s (row <- col):\n   ");
  for (int j = 0; j < ndev; ++j) std::printf(" %6d", j);
  std::printf("\n");
  for (int i = 0; i < ndev; ++i) {
    std::printf("%d: ", i);
    for (int j = 0; j < ndev; ++j) {
      if (i == j) {
        std::printf(" %6s", "-");
        continue;
      }
      HIP_CHECK(hipSetDevice(i));
      HIP_CHECK(hipMemcpyPeer(buf[i], i, buf[j], j, bytes));  // warm
      HIP_CHECK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < iters; ++k)
        HIP_CHECK(hipMemcpyPeer(buf[i], i, buf[j], j, bytes));
      HIP_CHECK(hipDeviceSynchronize());
      const double us =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
              .count() /
          iters;
      std::printf(" %6.1f", bytes / us / 1e3);
    }
    std::printf("\n");
  }

  // B: direct remote reads from a kernel (grid-stride, 8 MB payload)
  const size_t n = (8 << 20) / sizeof(float);
  std::printf("\nkernel remote-read bandwidth GB/s (row reads col):\n   ");
  for (int j = 0; j < ndev; ++j) std::printf(" %6d", j);
  std::printf("\n");
  for (int i = 0; i < ndev; ++i) {
    std::printf("%d: ", i);
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < ndev; ++j) {
      if (i == j) {
        std::printf(" %6s", "-");
        continue;
      }
      remote_read<<<208, 256>>>(buf[j], buf[i], n);  // warm
      HIP_CHECK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < iters; ++k) remote_read<<<208, 256>>>(buf[j], buf[i], n);
      HIP_CHECK(hipDeviceSynchronize());
      const hipError_t err = hipGetLastError();
      if (err != hipSuccess) {
        std::printf(" %6s", "ERR");
        hipGetLastError();
        continue;
      }
      const double us =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
              .count() /
          iters;
      std::printf(" %6.1f", n * sizeof(float) / us / 1e3);
    }
    std::printf("\n");
  }

  // correctness spot-check: 0x3f memset byte pattern -> float bits 0x3f3f3f3f
  HIP_CHECK(hipSetDevice(0));
  float v = 0;
  remote_read<<<1, 1>>>(buf[1], buf[0], 1);
  HIP_CHECK(hipMemcpy(&v, buf[0], 4, hipMemcpyDeviceToHost));
  std::printf("\ncorrectness: remote float * 2 = %f (want 1.489260)\n", v);

  for (int i = 0; i < ndev; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipFree(buf[i]));
  }
  return 0;
}
