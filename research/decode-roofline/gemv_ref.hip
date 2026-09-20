// Hand-rolled reference kernels for the decode-roofline study (research,
// user-directed 2026-09): what does the HBM actually deliver for
// (a) a pure streaming read, and (b) a simple split-K bf16 GEMV — the two
// bounds the rocBLAS S=1 gemm_ex path is judged against. Compiled by hipcc
// (gfx90a); host wrappers exposed raw-pointer style, mirroring the engine's
// kernels.hip pattern.
//
// GEMV problem: y[N] (f32) = W (bf16, row-major [N, K]) . x (bf16 [K]).
// Same math the engine's gemm_ex calls compute at S=1 (TN, bf16 in / f32
// accumulate out). Split-K: grid (ceil(N/8), ceil(K/CK)); one warp per
// output row, 8 warps per CTA; per-chunk partials to global, then a small
// combine kernel. Deliberately simple — this is a bandwidth probe, not a
// production kernel; correctness is spot-checked in bench.cpp.

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ float b2f(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// Pure streaming read: grid-stride over the buffer as uint32 (2 bf16 per
// load), accumulate garbage, write one float per thread (defeats DCE).
__global__ void k_read_bw(const uint32_t* __restrict__ w, float* __restrict__ sink,
                          size_t n32) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n32; i += stride)
    acc += __uint_as_float(w[i]);
  sink[threadIdx.x] = acc;
}

// Split-K GEMV phase 1: warp w (of 8 per CTA) owns output row
// n = blockIdx.x*8 + w over K-chunk [blockIdx.y*CK, ...). uint64 loads
// (4 bf16 per lane per step) along the row, warp shuffle reduction, one
// partial per (chunk, row). K, CK and k0 must be multiples of 4.
__global__ void k_gemv_partial(const unsigned short* __restrict__ W,
                               const unsigned short* __restrict__ x,
                               float* __restrict__ partials, uint32_t N, uint32_t K,
                               uint32_t CK) {
  const uint32_t n = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t k0 = blockIdx.y * CK, k1 = min(K, k0 + CK);
  const uint64_t* Wr = reinterpret_cast<const uint64_t*>(W + (size_t)n * K);
  const uint64_t* xr = reinterpret_cast<const uint64_t*>(x);
  float acc = 0.0f;
  for (uint32_t p = (k0 >> 2) + lane; p < (k1 >> 2); p += 32) {
    const uint64_t wv = Wr[p], xv = xr[p];
    acc += b2f((unsigned short)(wv & 0xffff)) * b2f((unsigned short)(xv & 0xffff));
    acc += b2f((unsigned short)((wv >> 16) & 0xffff)) *
           b2f((unsigned short)((xv >> 16) & 0xffff));
    acc += b2f((unsigned short)((wv >> 32) & 0xffff)) *
           b2f((unsigned short)((xv >> 32) & 0xffff));
    acc += b2f((unsigned short)(wv >> 48)) * b2f((unsigned short)(xv >> 48));
  }
  for (uint32_t off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffffffffffULL, acc, off);
  if (lane == 0) partials[(size_t)blockIdx.y * N + n] = acc;
}

// Phase 2: y[n] = sum of the C chunk partials. One thread per output.
__global__ void k_gemv_combine(const float* __restrict__ partials, float* __restrict__ y,
                               uint32_t N, uint32_t C) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.0f;
  for (uint32_t c = 0; c < C; ++c) a += partials[(size_t)c * N + n];
  y[n] = a;
}

unsigned grid_for(size_t n, unsigned block) {
  unsigned g = (unsigned)((n + block - 1) / block);
  return g == 0 ? 1 : (g > 65535 ? 65535 : g);
}

#define HIP_CHECK(x)                                                                  \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess)                                                             \
      throw std::runtime_error(std::string("gemv_ref: " #x " failed: ") +             \
                               hipGetErrorString(e_));                                \
  } while (0)

}  // namespace

namespace decode_roofline {

// Streams `bytes` from `w` (device) and reports nothing; call R times and
// time with events from the host. sink must hold >= 256 floats.
void read_bw(const void* w, size_t bytes, float* sink) {
  if (bytes % 4) throw std::runtime_error("gemv_ref: read_bw wants 4-byte-aligned size");
  k_read_bw<<<832, 256>>>(static_cast<const uint32_t*>(w), sink, bytes / 4);
  HIP_CHECK(hipGetLastError());
}

// y = W.x (bf16 W [N,K] row-major, bf16 x [K], f32 y [N]). partials must
// hold ceil(K/CK)*N floats; CK is the K-chunk (2048 works for all engine
// shapes). Two launches (partial + combine).
void gemv_splitk(const void* W, const void* x, float* y, float* partials, uint32_t N,
                 uint32_t K, uint32_t CK) {
  if (K % 4) throw std::runtime_error("gemv_ref: K must be a multiple of 4");
  if (CK % 4) throw std::runtime_error("gemv_ref: CK must be a multiple of 4");
  const uint32_t chunks = (K + CK - 1) / CK;
  if (chunks > 65535) throw std::runtime_error("gemv_ref: too many chunks");
  k_gemv_partial<<<dim3((N + 7) / 8, chunks), 256>>>(
      static_cast<const unsigned short*>(W), static_cast<const unsigned short*>(x),
      partials, N, K, CK);
  k_gemv_combine<<<grid_for(N, 256), 256>>>(partials, y, N, chunks);
  HIP_CHECK(hipGetLastError());
}

}  // namespace decode_roofline
