// gemv_70b_sweep.hip — chunk-count sweep for the engine's S=1 split-K GEMV
// (k_gemv_partial copied verbatim from src/blocks/gpu_rocm/kernels.hip) on
// the 70B TP=4 decode shapes, plus a padded-K gate+up variant to test the
// channel-lockstep hypothesis (16 KB row stride ≡ 0 mod 32 channels; a
// 16640 B stride rotates phase by 1 channel per row).
//
// Shapes (per rank, TP=4): QKV [10240x8192]->N=2560, O [8192x8192]->N=2048,
// gate+up [57344x8192]->N=14336, down [8192x28672]->N=8192,K=7168.
// Current engine picks chunks = ceil(4096/row_blocks) clamped to K/1024.
#include <cstdio>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

#define CK_ERR(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "HIP error %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

namespace {
__device__ __forceinline__ float b2f(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ float qbf16(float f) {
  return __uint_as_float(__float_as_uint(f) & 0xffff0000u);
}

// === verbatim copy of the engine kernel (kXF32=true path used by gate+up/QKV)
template <bool kXF32>
__global__ void k_gemv_partial(const unsigned short* __restrict__ W,
                               const void* __restrict__ x, float* __restrict__ partials,
                               uint32_t N, uint32_t K, uint32_t CK) {
  const uint32_t n = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t k0 = blockIdx.y * CK, k1 = min(K, k0 + CK);
  const uint64_t* Wr = reinterpret_cast<const uint64_t*>(W + (size_t)n * K);
  float acc = 0.0f;
  if (kXF32) {
    const float4* xr = reinterpret_cast<const float4*>(x);
    for (uint32_t p = (k0 >> 2) + lane; p < (k1 >> 2); p += 32) {
      const uint64_t wv = Wr[p];
      const float4 xv = xr[p];
      acc += b2f((unsigned short)(wv & 0xffff)) * qbf16(xv.x);
      acc += b2f((unsigned short)((wv >> 16) & 0xffff)) * qbf16(xv.y);
      acc += b2f((unsigned short)((wv >> 32) & 0xffff)) * qbf16(xv.z);
      acc += b2f((unsigned short)(wv >> 48)) * qbf16(xv.w);
    }
  } else {
    const uint64_t* xr = reinterpret_cast<const uint64_t*>(x);
    for (uint32_t p = (k0 >> 2) + lane; p < (k1 >> 2); p += 32) {
      const uint64_t wv = Wr[p], xv = xr[p];
      acc += b2f((unsigned short)(wv & 0xffff)) * b2f((unsigned short)(xv & 0xffff));
      acc += b2f((unsigned short)((wv >> 16) & 0xffff)) *
             b2f((unsigned short)((xv >> 16) & 0xffff));
      acc += b2f((unsigned short)((wv >> 32) & 0xffff)) *
             b2f((unsigned short)((xv >> 32) & 0xffff));
      acc += b2f((unsigned short)(wv >> 48)) * b2f((unsigned short)(xv >> 48));
    }
  }
  for (uint32_t off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffffffffffULL, acc, off);
  if (lane == 0) partials[(size_t)blockIdx.y * N + n] = acc;
}
__global__ void k_gemv_combine(const float* __restrict__ partials, float* __restrict__ y,
                               uint32_t N, uint32_t chunks) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.0f;
  for (uint32_t c = 0; c < chunks; ++c) a += partials[(size_t)c * N + n];
  y[n] = a;
}
}  // namespace

struct Shape {
  const char* name;
  uint32_t N, K;
  bool xf32;
  uint32_t cur_chunks;  // what the engine picks today
};

double bench(const unsigned short* W, const void* x, float* partials, float* y,
             uint32_t N, uint32_t K, bool xf32, uint32_t chunks, uint32_t iters,
             bool align256 = false) {
  uint32_t CK = (((K + chunks - 1) / chunks) + 3) & ~3u;
  if (align256) CK = ((K + chunks - 1) / chunks + 127) & ~127u;  // 256B-aligned starts
  const dim3 grid((N + 7) / 8, chunks);
  auto launch = [&] {
    if (xf32)
      k_gemv_partial<true><<<grid, 256>>>(W, x, partials, N, K, CK);
    else
      k_gemv_partial<false><<<grid, 256>>>(W, x, partials, N, K, CK);
    k_gemv_combine<<<(N + 255) / 256, 256>>>(partials, y, N, chunks);
  };
  for (int i = 0; i < 5; ++i) launch();
  CK_ERR(hipDeviceSynchronize());
  hipEvent_t t0, t1;
  CK_ERR(hipEventCreate(&t0)); CK_ERR(hipEventCreate(&t1));
  CK_ERR(hipEventRecord(t0));
  for (uint32_t i = 0; i < iters; ++i) launch();
  CK_ERR(hipEventRecord(t1));
  CK_ERR(hipEventSynchronize(t1));
  float ms = 0;
  CK_ERR(hipEventElapsedTime(&ms, t0, t1));
  return (double)N * K * 2 * iters / (ms / 1000.0) / 1e9;  // GB/s
}

int main() {
  CK_ERR(hipSetDevice(0));
  const uint32_t iters = 50;
  const Shape shapes[] = {
    {"qkv   N=2560 K=8192 (xf32)", 2560, 8192, true, 8},
    {"o     N=2048 K=8192 (bf16)", 2048, 8192, false, 8},
    {"gup   N=14336 K=8192 (xf32)", 14336, 8192, true, 3},
    {"down  N=8192 K=7168 (bf16)", 8192, 7168, false, 4},
  };
  const uint32_t chunk_sweep[] = {1, 2, 3, 4, 6, 8};

  // allocate for the biggest case: gup N=14336 K=8192 (235 MB); padded variant 16640B stride
  const uint32_t Nmax = 14336, Kmax = 8192, Kpad = 8320;
  unsigned short* W;
  CK_ERR(hipMalloc(&W, (size_t)Nmax * Kpad * 2));
  CK_ERR(hipMemset(W, 0x3c, (size_t)Nmax * Kpad * 2));
  float* x;
  CK_ERR(hipMalloc(&x, (size_t)Kpad * 4));
  CK_ERR(hipMemset(x, 0, (size_t)Kpad * 4));
  float *partials, *y;
  CK_ERR(hipMalloc(&partials, (size_t)16 * Nmax * 4));
  CK_ERR(hipMalloc(&y, (size_t)Nmax * 4));
  CK_ERR(hipDeviceSynchronize());

  for (const auto& s : shapes) {
    printf("%s:\n", s.name);
    for (uint32_t ch : chunk_sweep) {
      if (ch > s.K / 1024) continue;  // engine's max_chunks rule
      double bw = bench(W, x, partials, y, s.N, s.K, s.xf32, ch, iters);
      printf("  chunks=%u CK=%-5u %s %8.1f GB/s%s\n", ch,
             (((s.K + ch - 1) / ch) + 3) & ~3u, "", bw,
             ch == s.cur_chunks ? "   <-- engine today" : "");
      if (bw < 950.0) {  // retry misaligned cases with 256B-aligned CK
        double bw2 = bench(W, x, partials, y, s.N, s.K, s.xf32, ch, iters, true);
        printf("  chunks=%u CK=%-5u %s %8.1f GB/s   (256B-aligned CK)\n", ch,
               ((s.K + ch - 1) / ch + 127) & ~127u, "", bw2);
      }
    }
  }

  // padded-K gate+up: row stride 8320*2 = 16640 B (≡ 65 mod 32 -> phase rotates)
  printf("gup PADDED K=8320 (stride 16640 B, lockstep broken):\n");
  for (uint32_t ch : chunk_sweep) {
    if (ch > Kpad / 1024) continue;
    double bw = bench(W, x, partials, y, 14336, Kpad, true, ch, iters);
    // report effective bandwidth on USEFUL bytes (K=8192 of each row)
    double bw_useful = bw * 8192.0 / Kpad;
    printf("  chunks=%u  raw %8.1f GB/s  (useful-byte adj %8.1f GB/s)\n", ch, bw, bw_useful);
  }
  return 0;
}
