// gemv_naive.hip — naive thread-per-row GEMV study (user design):
//   total threads = output channels of the matrix (one per row, no split-K,
//   no shuffle/wave reduction), blocks of 64 (native wave size), 4 FMA
//   accumulators to break the dependency chain (v4-attention style).
// Each thread streams its ENTIRE row (16 KB at K=8192) sequentially; a wave's
// 64 lanes read addresses exactly one row (16 KB) apart at every step — so
// the 64 lanes map to the same 256B-stripe phase unless the hardware XORs
// HIGH address bits into the channel select. Measured bandwidth therefore
// answers: "does the hardware swizzle enable a naive implementation?"
//
// Variants: block size {64,128,256}; shapes {gup, down, qkv, o}; padded-stride
// gup (16640 B rows); a pure-read version (no x, no math) to separate the
// access pattern's raw bandwidth from the FMA path. Correctness vs host ref.
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
__device__ __forceinline__ float dot4(uint64_t w, float4 xv) {
  float a = b2f((unsigned short)(w & 0xffff)) * qbf16(xv.x);
  a += b2f((unsigned short)((w >> 16) & 0xffff)) * qbf16(xv.y);
  a += b2f((unsigned short)((w >> 32) & 0xffff)) * qbf16(xv.z);
  a += b2f((unsigned short)(w >> 48)) * qbf16(xv.w);
  return a;
}

// naive thread-per-row: 4 accumulators, 4 uint64 (16 bf16) per iteration
__global__ void k_gemv_naive(const unsigned short* __restrict__ W,
                             const float* __restrict__ x, float* __restrict__ y,
                             uint32_t N, uint32_t K) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const uint64_t* Wr = reinterpret_cast<const uint64_t*>(W + (size_t)n * K);
  const float4* xr = reinterpret_cast<const float4*>(x);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  const uint32_t steps = K >> 2;
  for (uint32_t p = 0; p < steps; p += 4) {
    const uint64_t w0 = Wr[p], w1 = Wr[p + 1], w2 = Wr[p + 2], w3 = Wr[p + 3];
    const float4 x0 = xr[p], x1 = xr[p + 1], x2 = xr[p + 2], x3 = xr[p + 3];
    a0 += dot4(w0, x0);
    a1 += dot4(w1, x1);
    a2 += dot4(w2, x2);
    a3 += dot4(w3, x3);
  }
  y[n] = a0 + a1 + a2 + a3;
}

// pure-read twin: same access pattern, no x/math — raw pattern bandwidth
__global__ void k_read_naive(const unsigned short* __restrict__ W,
                             float* __restrict__ y, uint32_t N, uint32_t K) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const uint64_t* Wr = reinterpret_cast<const uint64_t*>(W + (size_t)n * K);
  uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  const uint32_t steps = K >> 2;
  for (uint32_t p = 0; p < steps; p += 4) {
    s0 += Wr[p]; s1 += Wr[p + 1]; s2 += Wr[p + 2]; s3 += Wr[p + 3];
  }
  if ((s0 + s1 + s2 + s3) == 0xdeadbeefcafebabeULL) y[n] = 1.0f;  // never
}
}  // namespace

double bench(const unsigned short* W, const float* x, float* y, uint32_t N,
             uint32_t K, uint32_t block, bool pure, uint32_t iters) {
  const uint32_t grid = (N + block - 1) / block;
  auto launch = [&] {
    if (pure) k_read_naive<<<grid, block>>>(W, y, N, K);
    else k_gemv_naive<<<grid, block>>>(W, x, y, N, K);
  };
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
  return (double)N * K * 2 * iters / (ms / 1000.0) / 1e9;
}

int main() {
  CK(hipSetDevice(0));
  const uint32_t iters = 50;
  const uint32_t Nmax = 14336, Kmax = 8192, Kpad = 8320;

  // host-init W with a checkable pattern: w[i] = bf16(i % 13), x[i] = i % 7
  std::vector<unsigned short> hw((size_t)Nmax * Kpad);
  for (size_t i = 0; i < hw.size(); ++i) {
    float f = float(i % 13);
    uint32_t bits;
    memcpy(&bits, &f, 4);
    hw[i] = (unsigned short)(bits >> 16);  // proper bf16 bit pattern
  }
  std::vector<float> hx(Kpad);
  for (uint32_t i = 0; i < Kpad; ++i) hx[i] = float(i % 7);
  unsigned short* W; CK(hipMalloc(&W, hw.size() * 2));
  CK(hipMemcpy(W, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
  float* x; CK(hipMalloc(&x, hx.size() * 4));
  CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  float* y; CK(hipMalloc(&y, Nmax * 4));

  struct Shape { const char* name; uint32_t N, K; } shapes[] = {
    {"gup  N=14336 K=8192", 14336, 8192},
    {"down N=8192  K=7168", 8192, 7168},
    {"qkv  N=2560  K=8192", 2560, 8192},
    {"o    N=2048  K=8192", 2048, 8192},
  };
  const uint32_t blocks[] = {64, 128, 256};

  for (const auto& s : shapes) {
    printf("%s:\n", s.name);
    for (uint32_t b : blocks) {
      double bw = bench(W, x, y, s.N, s.K, b, false, iters);
      printf("  block=%-4u gemv %8.1f GB/s", b, bw);
      if (b == 64) {
        double pr = bench(W, x, y, s.N, s.K, b, true, iters);
        printf("   pure-read %8.1f GB/s", pr);
      }
      printf("\n");
    }
    // correctness spot-check at block=64: y[n] = sum_k bf16(k%13) * bf16(k%7)
    // constant per row: sum over K of ((k%13)<<8 as bf16) * (k%7 truncated)
    const uint32_t grid = (s.N + 63) / 64;
    k_gemv_naive<<<grid, 64>>>(W, x, y, s.N, s.K);
    CK(hipDeviceSynchronize());
    std::vector<float> hy(4);
    CK(hipMemcpy(hy.data(), y, 16, hipMemcpyDeviceToHost));
    double expect = 0;
    for (uint32_t k = 0; k < s.K; ++k) {
      float xk = float(k % 7);
      uint32_t bits;
      memcpy(&bits, &xk, 4);
      bits &= 0xffff0000u;  // qbf16 truncation, host-side
      float xq;
      memcpy(&xq, &bits, 4);
      expect += double(k % 13) * double(xq);
    }
    printf("  check y[0]=%.1f expect=%.1f %s\n", hy[0], expect,
           fabs(hy[0] - expect) < 1.0 ? "OK" : "FAIL");
  }

  printf("gup PADDED stride 16640 B:\n");
  for (uint32_t b : blocks) {
    double bw = bench(W, x, y, 14336, Kpad, b, false, iters);
    printf("  block=%-4u gemv %8.1f GB/s (useful-adj %8.1f)\n", b, bw, bw * 8192.0 / Kpad);
  }
  printf("(reference: engine split-K on gup = 772 (CK2732) / 1017 (CK2816) GB/s;\n"
         " measured read ceiling ~1290 GB/s; padded-stride split-K chunks=1 ~1193)\n");
  return 0;
}
