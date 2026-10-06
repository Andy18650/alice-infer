// gemv_cm_sk.hip — packed ("almost column-major") naive GEMV + split-K.
// User direction: gup's 14336 threads = half the 28160-thread GCD fill
// (110 CU x 4 SIMD x 64), so split K into chunks of {256, 512, 1024}
// elements computed in parallel (threads = N * K/CW), partials written out
// without a gather (the question is "can this pattern saturate VRAM
// bandwidth?"); a small combine kernel is included for correctness only.
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

// one thread per (row, K-chunk); uint32 (2 bf16) per step from the packed
// matrix; partials[c*N + n] = row n's sum over chunk c. No shuffle.
__global__ void k_gemv_cm_sk(const uint32_t* __restrict__ Wp,
                             const float* __restrict__ x,
                             float* __restrict__ partials, uint32_t N, uint32_t K,
                             uint32_t CW) {
  const uint32_t chunks = K / CW;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * chunks) return;
  const uint32_t n = idx % N;
  const uint32_t c = idx / N;
  const uint32_t p0 = c * (CW >> 1), p1 = p0 + (CW >> 1);
  const float4* xr = reinterpret_cast<const float4*>(x);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (uint32_t p = p0; p < p1; p += 4) {
    const uint32_t w0 = Wp[(size_t)p * N + n];
    const uint32_t w1 = Wp[(size_t)(p + 1) * N + n];
    const uint32_t w2 = Wp[(size_t)(p + 2) * N + n];
    const uint32_t w3 = Wp[(size_t)(p + 3) * N + n];
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
  partials[(size_t)c * N + n] = a0 + a1 + a2 + a3;
}

// correctness only: y[n] = sum_c partials[c*N + n]
__global__ void k_combine(const float* __restrict__ partials, float* __restrict__ y,
                          uint32_t N, uint32_t chunks) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.f;
  for (uint32_t c = 0; c < chunks; ++c) a += partials[(size_t)c * N + n];
  y[n] = a;
}
}  // namespace

int main() {
  CK(hipSetDevice(0));
  const uint32_t iters = 50;
  const uint32_t Nmax = 14336, Kmax = 8192;

  std::vector<float> hx(Kmax);
  for (uint32_t i = 0; i < Kmax; ++i) hx[i] = float(i % 7);
  uint32_t* Wp;
  CK(hipMalloc(&Wp, (size_t)Nmax * (Kmax / 2) * 4));
  float* x;
  CK(hipMalloc(&x, hx.size() * 4));
  CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  float *partials, *y;
  CK(hipMalloc(&partials, (size_t)32 * Nmax * 4));
  CK(hipMalloc(&y, Nmax * 4));

  struct Shape { const char* name; uint32_t N, K; } shapes[] = {
    {"gup  N=14336 K=8192", 14336, 8192},
    {"down N=8192  K=7168", 8192, 7168},
    {"qkv  N=2560  K=8192", 2560, 8192},
    {"o    N=2048  K=8192", 2048, 8192},
  };
  const uint32_t cws[] = {256, 512, 1024};

  for (const auto& s : shapes) {
    // pack with this shape's N
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
    for (uint32_t cw : cws) {
      if (s.K % cw) { printf("  CW=%-5u (skipped, K not divisible)\n", cw); continue; }
      const uint32_t chunks = s.K / cw;
      const uint32_t threads = s.N * chunks;
      const uint32_t grid = (threads + 63) / 64;
      auto launch = [&] { k_gemv_cm_sk<<<grid, 64>>>(Wp, x, partials, s.N, s.K, cw); };
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
      printf("  CW=%-5u threads=%-7u chunks=%-3u %8.1f GB/s\n", cw, threads, chunks, bw);
      // correctness (row 0): combine and compare
      k_gemv_cm_sk<<<grid, 64>>>(Wp, x, partials, s.N, s.K, cw);
      k_combine<<<(s.N + 255) / 256, 256>>>(partials, y, s.N, chunks);
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
      printf("       check y[0]=%.1f expect=%.1f %s\n", y0, expect,
             fabs(y0 - expect) < 1.0 ? "OK" : "FAIL");
    }
  }
  printf("(reference: packed no-split gup 507; split-K warp-per-row 1017; ceiling ~1290)\n");
  return 0;
}
