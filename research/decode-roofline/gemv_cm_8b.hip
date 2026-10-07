// col-major engine kernel, all 8B tp=1 shapes, fp32 and bf16 x variants
#include <cstdio>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
namespace {
__device__ __forceinline__ float b2f(unsigned short b){return __uint_as_float((unsigned int)b<<16);}
__device__ __forceinline__ float qbf16(float f){return __uint_as_float(__float_as_uint(f)&0xffff0000u);}
template <bool kXF32>
__global__ void k_cm(const uint32_t* __restrict__ Wc, const void* __restrict__ x,
                     float* __restrict__ partials, uint32_t N, uint32_t K) {
  const uint32_t pairs_n = N >> 1;
  const uint32_t chunks = (K + 255) / 256;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pairs_n * chunks) return;
  const uint32_t t = idx % pairs_n, c = idx / pairs_n;
  const uint32_t k0 = c * 256, k1 = min(K, k0 + 256);
  float a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
  if (kXF32) {
    const float* xf = (const float*)x;
    for (uint32_t k = k0; k < k1; k += 4) {
      uint32_t w0=Wc[(size_t)k*pairs_n+t], w1=Wc[(size_t)(k+1)*pairs_n+t],
               w2=Wc[(size_t)(k+2)*pairs_n+t], w3=Wc[(size_t)(k+3)*pairs_n+t];
      float x0=qbf16(xf[k]),x1=qbf16(xf[k+1]),x2=qbf16(xf[k+2]),x3=qbf16(xf[k+3]);
      a0+=b2f((unsigned short)(w0&0xffff))*x0; b0+=b2f((unsigned short)(w0>>16))*x0;
      a1+=b2f((unsigned short)(w1&0xffff))*x1; b1+=b2f((unsigned short)(w1>>16))*x1;
      a2+=b2f((unsigned short)(w2&0xffff))*x2; b2+=b2f((unsigned short)(w2>>16))*x2;
      a3+=b2f((unsigned short)(w3&0xffff))*x3; b3+=b2f((unsigned short)(w3>>16))*x3;
    }
  } else {
    const unsigned short* xb = (const unsigned short*)x;
    for (uint32_t k = k0; k < k1; k += 4) {
      uint32_t w0=Wc[(size_t)k*pairs_n+t], w1=Wc[(size_t)(k+1)*pairs_n+t],
               w2=Wc[(size_t)(k+2)*pairs_n+t], w3=Wc[(size_t)(k+3)*pairs_n+t];
      float x0=b2f(xb[k]),x1=b2f(xb[k+1]),x2=b2f(xb[k+2]),x3=b2f(xb[k+3]);
      a0+=b2f((unsigned short)(w0&0xffff))*x0; b0+=b2f((unsigned short)(w0>>16))*x0;
      a1+=b2f((unsigned short)(w1&0xffff))*x1; b1+=b2f((unsigned short)(w1>>16))*x1;
      a2+=b2f((unsigned short)(w2&0xffff))*x2; b2+=b2f((unsigned short)(w2>>16))*x2;
      a3+=b2f((unsigned short)(w3&0xffff))*x3; b3+=b2f((unsigned short)(w3>>16))*x3;
    }
  }
  partials[(size_t)c*N+2*t] = a0+a1+a2+a3;
  partials[(size_t)c*N+2*t+1] = b0+b1+b2+b3;
}
}
double bench(const uint32_t* W, const void* x, float* p, uint32_t N, uint32_t K,
             bool xf32, uint32_t iters) {
  const uint32_t threads = (N/2)*((K+255)/256);
  const uint32_t grid = (threads + 63) / 64;
  auto launch = [&]{ if (xf32) k_cm<true><<<grid,64>>>(W,x,p,N,K); else k_cm<false><<<grid,64>>>(W,x,p,N,K); };
  for (int i=0;i<5;++i) launch();
  CK(hipDeviceSynchronize());
  hipEvent_t t0,t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
  CK(hipEventRecord(t0));
  for (uint32_t i=0;i<iters;++i) launch();
  CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
  float ms=0; CK(hipEventElapsedTime(&ms,t0,t1));
  return (double)N*K*2*iters/(ms/1000.0)/1e9;
}
int main() {
  CK(hipSetDevice(0));
  const uint32_t iters = 50;
  struct S { const char* name; uint32_t N, K; bool xf32; };
  S shapes[] = {
    {"gup  N=28672 K=4096  xf32", 28672, 4096, true},
    {"down N=4096  K=14336 bf16", 4096, 14336, false},
    {"qkv  N=6144  K=4096  xf32", 6144, 4096, true},
    {"o    N=4096  K=4096  bf16", 4096, 4096, false},
    {"70B-gup N=14336 K=8192 xf32", 14336, 8192, true},
  };
  for (auto& s : shapes) {
    std::vector<uint32_t> hw((size_t)s.N*s.K/2, 0x3f803f80u);
    std::vector<float> hx(s.K, 1.0f);
    uint32_t* W; CK(hipMalloc(&W, hw.size()*4)); CK(hipMemcpy(W,hw.data(),hw.size()*4,hipMemcpyHostToDevice));
    float* x; CK(hipMalloc(&x, s.K*4)); CK(hipMemcpy(x,hx.data(),s.K*4,hipMemcpyHostToDevice));
    float* p; CK(hipMalloc(&p, (size_t)64*s.N*4));
    double bw = bench(W,x,p,s.N,s.K,s.xf32,iters);
    printf("%-32s %8.1f GB/s\n", s.name, bw);
    hipFree(W); hipFree(x); hipFree(p);
  }
  return 0;
}
