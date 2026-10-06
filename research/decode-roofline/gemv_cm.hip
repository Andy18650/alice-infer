// gemv_cm.hip — naive thread-per-row GEMV on an "almost column-major" packed
// matrix (user design): store column-PAIRS interleaved across rows,
//   Wp[p*N + n] = (W[n][2p+1] << 16) | W[n][2p]     (one uint32 = 2 bf16)
// so thread n (row n) at step p loads Wp[p*N + n] — a wave's 64 lanes then
// read a contiguous 64*4 = 256 B range per step (fully coalesced), and
// adjacent waves read adjacent ranges. Same naive kernel otherwise: one
// thread per row, 4 FMA accumulators, no split-K, no shuffle.
// If this does not yield a significant improvement over the row-major naive
// study (gemv_naive.hip: gup 398 GB/s), the problem is probably NOT
// coalescing.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "HIP error %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

namespace {
__device__ __forceinline__ float b2f(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ float qbf16(float f) {
  return __uint_as_float(__float_as_uint(f) & 0xffff0000u);
}

// one thread per row; uint32 (2 bf16) per step from the pair-packed matrix.
// ld = pack stride (= N of the packed copy).
__global__ void k_gemv_cm(const uint32_t* __restrict__ Wp,
                          const float* __restrict__ x, float* __restrict__ y,
                          uint32_t N, uint32_t K, uint32_t ld) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const uint32_t pairs = K >> 1;
  const float4* xr = reinterpret_cast<const float4*>(x);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (uint32_t p = 0; p < pairs; p += 4) {
    const uint32_t w0 = Wp[(size_t)p * ld + n];
    const uint32_t w1 = Wp[(size_t)(p + 1) * ld + n];
    const uint32_t w2 = Wp[(size_t)(p + 2) * ld + n];
    const uint32_t w3 = Wp[(size_t)(p + 3) * ld + n];
    const float4 xa = xr[p >> 1], xb = xr[(p >> 1) + 1];
    a0 += b2f((unsigned short)(w0 & 0xffff)) * qbf16(xa.x) +
          b2f((unsigned short)(w0 >> 16)) * qbf16(xa.y);
    a1 += b2f((unsigned short)(w1 & 0xffff)) * qbf16(xa.z) +
          b2f((unsigned short)(w1 >> 16)) * qbf16(xa.w);
    a2 += b2f((unsigned short)(w2 & 0xffff)) * qbf16(xb.x) +
          b2f((unsigned short)(w2 >> 16)) * qbf16(xb.y);
    a3 += b2f((unsigned short)(w3 & 0xffff)) * qbf16(xb.z) +
          b2f((unsigned short)(w3 >> 16)) * qbf16(xb.w);
  }
  y[n] = a0 + a1 + a2 + a3;
}
}  // namespace

int main() {
  CK(hipSetDevice(0));
  const uint32_t iters = 50;
  const uint32_t Nmax = 14336, Kmax = 8192;

  // host pattern: element (n,k) = bf16(k%13) (all rows identical), x[i]=i%7
  std::vector<float> hx(Kmax);
  for (uint32_t i = 0; i < Kmax; ++i) hx[i] = float(i % 7);
  uint32_t* Wp;
  CK(hipMalloc(&Wp, (size_t)Nmax * (Kmax / 2) * 4));
  float* x;
  CK(hipMalloc(&x, hx.size() * 4));
  CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  float* y;
  CK(hipMalloc(&y, Nmax * 4));

  struct Shape { const char* name; uint32_t N, K; } shapes[] = {
    {"gup  N=14336 K=8192", 14336, 8192},
    {"down N=8192  K=7168", 8192, 7168},
    {"qkv  N=2560  K=8192", 2560, 8192},
    {"o    N=2048  K=8192", 2048, 8192},
  };

  for (const auto& s : shapes) {
    // pack with THIS shape's N: Wp[p*N + n] = bf16(2p%13) | bf16((2p+1)%13)<<16
    std::vector<uint32_t> hwp((size_t)s.N * (s.K / 2));
    for (uint32_t p = 0; p < s.K / 2; ++p) {
      uint32_t lo, hi, bits;
      float f0 = float((2 * p) % 13), f1 = float((2 * p + 1) % 13);
      memcpy(&bits, &f0, 4); lo = bits >> 16;
      memcpy(&bits, &f1, 4); hi = bits >> 16;
      for (uint32_t n = 0; n < s.N; ++n)
        hwp[(size_t)p * s.N + n] = lo | (hi << 16);
    }
    CK(hipMemcpy(Wp, hwp.data(), hwp.size() * 4, hipMemcpyHostToDevice));
    printf("%s:\n", s.name);
    for (uint32_t b : {64u, 128u, 256u}) {
      const uint32_t grid = (s.N + b - 1) / b;
      auto launch = [&] { k_gemv_cm<<<grid, b>>>(Wp, x, y, s.N, s.K, s.N); };
      for (int i = 0; i < 5; ++i) launch();
      CK(hipDeviceSynchronize());
      hipEvent_t t0, t1;
      CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
      CK(hipEventRecord(t0));
      for (uint32_t i = 0; i < iters; ++i) launch();
      CK(hipEventRecord(t1));
      CK(hipEventSynchronize(t1));
      float ms = 0;
      CK(hipEventElapsedTime(&ms, t0, t1));
      double bw = (double)s.N * s.K * 2 * iters / (ms / 1000.0) / 1e9;
      printf("  block=%-4u gemv_cm %8.1f GB/s\n", b, bw);
    }
    // correctness: expected per-row sum (same as gemv_naive)
    const uint32_t grid = (s.N + 63) / 64;
    k_gemv_cm<<<grid, 64>>>(Wp, x, y, s.N, s.K, s.N);
    CK(hipDeviceSynchronize());
    float y0;
    CK(hipMemcpy(&y0, y, 4, hipMemcpyDeviceToHost));
    double expect = 0;
    for (uint32_t k = 0; k < s.K; ++k) {
      float xk = float(k % 7);
      uint32_t bits;
      memcpy(&bits, &xk, 4);
      bits &= 0xffff0000u;
      float xq;
      memcpy(&xq, &bits, 4);
      expect += double(k % 13) * double(xq);
    }
    printf("  check y[0]=%.1f expect=%.1f %s\n", y0, expect,
           fabs(y0 - expect) < 1.0 ? "OK" : "FAIL");
  }
  printf("(reference: row-major naive gup 398 GB/s; split-K aligned 1017; ceiling ~1290)\n");
  return 0;
}
