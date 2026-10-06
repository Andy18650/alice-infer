// l2pool.hip — is the gfx90a GCD's 8 MiB L2 logically POOLED (any CU can hit
// any slice at L2 latency), or SE-private (a CU hits only its own SE's ~2 MiB
// slice, rest falls to HBM)?
//
// Method: pointer-chase (fully serialized dependent loads => true load-to-use
// latency, no MLP) from ONE block on ONE CU.
//   case A: 2 MiB buffer, L2-warmed  -> pure L2 latency baseline
//   case B: 8 MiB buffer, L2-warmed  -> pooled? (== A) or private? (~0.25*A+0.75*C)
//   case C: 8 MiB after 64 MiB eviction -> HBM latency baseline
// The chase step (4097 words, odd) is coprime with the buffer size, so the
// chain samples addresses uniformly across ALL L2 slices.
#include <cstdio>
#include <vector>
#include <hip/hip_runtime.h>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void k_warm(const float4* __restrict__ buf, size_t n4, float* sink) {
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 v = buf[i];
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  if (s.x == 12345.678f) sink[0] = s.x;  // never true; keeps the loads alive
}

__global__ void k_chase(volatile const unsigned int* buf, unsigned long long steps,
                        unsigned long long* ticks_out, unsigned int* idx_out) {
  unsigned int idx = 0;
  unsigned long long t0 = wall_clock64();
  for (unsigned long long k = 0; k < steps; ++k) idx = buf[idx];  // dependent chain
  unsigned long long t1 = wall_clock64();
  ticks_out[0] = t1 - t0;
  idx_out[0] = idx;
}

static unsigned int* make_chase_buf(int dev, size_t words, unsigned int /*step*/) {
  // Random Hamiltonian cycle: buf[perm[k]] = perm[k+1], so the pointer chase
  // walks addresses in a random, unpredictable order (defeats any stride
  // prefetcher). Fisher-Yates with xorshift64.
  std::vector<unsigned int> perm(words), h(words);
  for (size_t i = 0; i < words; ++i) perm[i] = (unsigned int)i;
  unsigned long long s = 0x9E3779B97F4A7C15ull;
  for (size_t i = words - 1; i > 0; --i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    size_t j = (size_t)(s % (i + 1));
    unsigned int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (size_t k = 0; k < words; ++k) h[perm[k]] = perm[(k + 1) % words];
  unsigned int* d;
  HIP_CHECK(hipSetDevice(dev));
  HIP_CHECK(hipMalloc(&d, words * 4));
  HIP_CHECK(hipMemcpy(d, h.data(), words * 4, hipMemcpyHostToDevice));
  return d;
}

static double chase_ns(int dev, const unsigned int* buf, unsigned long long steps) {
  HIP_CHECK(hipSetDevice(dev));
  unsigned long long* t; unsigned int* o;
  HIP_CHECK(hipMalloc(&t, 8)); HIP_CHECK(hipMalloc(&o, 4));
  k_chase<<<1, 1>>>(buf, steps, t, o);
  HIP_CHECK(hipDeviceSynchronize());
  unsigned long long ticks;
  HIP_CHECK(hipMemcpy(&ticks, t, 8, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(t)); HIP_CHECK(hipFree(o));
  return ticks * 40.0 / (double)steps;  // 25 MHz clock -> 40 ns/tick
}

int main() {
  const int dev = 0;
  const unsigned long long steps = 200000;
  HIP_CHECK(hipSetDevice(dev));
  float* sink; HIP_CHECK(hipMalloc(&sink, 4)); HIP_CHECK(hipMemset(sink, 0, 4));

  unsigned int* buf1m  = make_chase_buf(dev, 1u << 18, 4097);   // 1 MiB (L2 baseline)
  unsigned int* buf4m  = make_chase_buf(dev, 1u << 20, 4097);   // 4 MiB (pooled: all hit; private: 50%)
  unsigned int* buf6m  = make_chase_buf(dev, 3u << 19, 4097);   // 6 MiB (pooled: all hit; private: 33%)
  unsigned int* buf8m  = make_chase_buf(dev, 1u << 21, 4097);   // 8 MiB (whole L2)
  unsigned int* buf64m = make_chase_buf(dev, 1u << 24, 4097);   // 64 MiB (evictor)
  HIP_CHECK(hipDeviceSynchronize());
  // note: 6 MiB buffer is 3*2^18 words (not pow2); the cycle wraps via % words
  // in make_chase_buf's perm construction, which handles non-pow2 fine.

  auto measure = [&](const char* label, const unsigned int* chase, size_t chase_bytes,
                     const float4* warm, size_t warm_bytes) {
    if (warm) { k_warm<<<220, 1024>>>(warm, warm_bytes / 16, sink); HIP_CHECK(hipDeviceSynchronize()); }
    double ns1 = chase_ns(dev, chase, steps);
    double ns2 = chase_ns(dev, chase, steps);  // second run: self-warmed, gauges noise
    printf("%-40s %7.1f  (rerun %7.1f) ns/load\n", label, ns1, ns2);
    return ns1;
  };

  double a  = measure("A: 1 MiB warm (L2 baseline)", buf1m, 1u << 20, (const float4*)buf1m, 1u << 20);
  double b4 = measure("B4: 4 MiB warm", buf4m, 4u << 20, (const float4*)buf4m, 4u << 20);
  double b6 = measure("B6: 6 MiB warm", buf6m, 6u << 20, (const float4*)buf6m, 6u << 20);
  double b8 = measure("B8: 8 MiB warm", buf8m, 8u << 20, (const float4*)buf8m, 8u << 20);
  k_warm<<<220, 1024>>>((const float4*)buf64m, (1u << 24) / 4, sink);
  HIP_CHECK(hipDeviceSynchronize());
  double c  = measure("C: 8 MiB evicted (HBM baseline)", buf8m, 8u << 20, nullptr, 0);

  printf("\nif POOLED:   B4 ~= B6 ~= A (all resident at L2 latency)\n");
  printf("if PRIVATE:  B4 ~= 0.5*A+0.5*C = %.1f,  B6 ~= 0.33*A+0.67*C = %.1f\n",
         0.5 * a + 0.5 * c, 0.33 * a + 0.67 * c);
  printf("measured: A=%.1f B4=%.1f B6=%.1f B8=%.1f C=%.1f\n", a, b4, b6, b8, c);
  return 0;
}
