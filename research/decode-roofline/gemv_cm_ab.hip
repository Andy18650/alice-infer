// gemv_cm_ab.hip — head-to-head: pair-pack layout (study [10], 1277 GB/s)
// vs pure col-major layout (engine), same thread-per-(pair,chunk) scheme.
#include <cstdio>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
namespace {
__device__ __forceinline__ float b2f(unsigned short b){return __uint_as_float((unsigned int)b<<16);}
__device__ __forceinline__ float qbf16(float f){return __uint_as_float(__float_as_uint(f)&0xffff0000u);}

// A: pair-pack (study [10]): thread n, chunk of pairs; uint32 = (row n, cols 2p,2p+1)
__global__ void k_pp(const uint32_t* __restrict__ Wp, const float* __restrict__ x,
                     float* __restrict__ partials, uint32_t N, uint32_t K, uint32_t CW) {
  const uint32_t chunks = K / CW;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * chunks) return;
  const uint32_t n = idx % N, c = idx / N;
  const uint32_t p0 = c * (CW >> 1), p1 = p0 + (CW >> 1);
  const float4* xr = (const float4*)x;
  float a0=0,a1=0,a2=0,a3=0;
  for (uint32_t p = p0; p < p1; p += 4) {
    uint32_t w0=Wp[(size_t)p*N+n], w1=Wp[(size_t)(p+1)*N+n], w2=Wp[(size_t)(p+2)*N+n], w3=Wp[(size_t)(p+3)*N+n];
    float4 xa=xr[p>>1], xb=xr[(p>>1)+1];
    a0 += b2f((unsigned short)(w0&0xffff))*qbf16(xa.x) + b2f((unsigned short)(w0>>16))*qbf16(xa.y);
    a1 += b2f((unsigned short)(w1&0xffff))*qbf16(xa.z) + b2f((unsigned short)(w1>>16))*qbf16(xa.w);
    a2 += b2f((unsigned short)(w2&0xffff))*qbf16(xb.x) + b2f((unsigned short)(w2>>16))*qbf16(xb.y);
    a3 += b2f((unsigned short)(w3&0xffff))*qbf16(xb.z) + b2f((unsigned short)(w3>>16))*qbf16(xb.w);
  }
  partials[(size_t)c*N+n] = a0+a1+a2+a3;
}
// B: pure col-major (engine): thread t (rows 2t,2t+1), chunk of K; uint32 = (rows 2t,2t+1, col k)
__global__ void k_cm(const uint32_t* __restrict__ Wc, const float* __restrict__ x,
                     float* __restrict__ partials, uint32_t N, uint32_t K, uint32_t CW) {
  const uint32_t pairs_n = N >> 1;
  const uint32_t chunks = (K + CW - 1) / CW;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pairs_n * chunks) return;
  const uint32_t t = idx % pairs_n, c = idx / pairs_n;
  const uint32_t k0 = c * CW, k1 = min(K, k0 + CW);
  float a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
  for (uint32_t k = k0; k < k1; k += 4) {
    uint32_t w0=Wc[(size_t)k*pairs_n+t], w1=Wc[(size_t)(k+1)*pairs_n+t],
             w2=Wc[(size_t)(k+2)*pairs_n+t], w3=Wc[(size_t)(k+3)*pairs_n+t];
    float x0=qbf16(x[k]),x1=qbf16(x[k+1]),x2=qbf16(x[k+2]),x3=qbf16(x[k+3]);
    a0 += b2f((unsigned short)(w0&0xffff))*x0; b0 += b2f((unsigned short)(w0>>16))*x0;
    a1 += b2f((unsigned short)(w1&0xffff))*x1; b1 += b2f((unsigned short)(w1>>16))*x1;
    a2 += b2f((unsigned short)(w2&0xffff))*x2; b2 += b2f((unsigned short)(w2>>16))*x2;
    a3 += b2f((unsigned short)(w3&0xffff))*x3; b3 += b2f((unsigned short)(w3>>16))*x3;
  }
  partials[(size_t)c*N+2*t] = a0+a1+a2+a3;
  partials[(size_t)c*N+2*t+1] = b0+b1+b2+b3;
}
}
double bench(const uint32_t* W, const float* x, float* p, uint32_t N, uint32_t K,
             uint32_t CW, bool colmajor, uint32_t iters) {
  const uint32_t threads = colmajor ? (N/2)*((K+CW-1)/CW) : N*(K/CW);
  const uint32_t grid = (threads + 63) / 64;
  auto launch = [&]{ if (colmajor) k_cm<<<grid,64>>>(W,x,p,N,K,CW); else k_pp<<<grid,64>>>(W,x,p,N,K,CW); };
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
  const uint32_t N=14336, K=8192, iters=50;
  std::vector<uint32_t> hw((size_t)N*K/2);
  for (size_t i=0;i<hw.size();++i) hw[i]=0x3f803f80u;  // bf16(1.0) pairs
  std::vector<float> hx(K, 1.0f);
  uint32_t* W; CK(hipMalloc(&W, hw.size()*4)); CK(hipMemcpy(W,hw.data(),hw.size()*4,hipMemcpyHostToDevice));
  float* x; CK(hipMalloc(&x, K*4)); CK(hipMemcpy(x,hx.data(),K*4,hipMemcpyHostToDevice));
  float* p; CK(hipMalloc(&p, (size_t)64*N*4));
  for (uint32_t cw : {256u, 128u, 64u}) {
    double a = bench(W,x,p,N,K,cw,false,iters);
    double b = bench(W,x,p,N,K,cw,true,iters);
    printf("CW=%-4u pair-pack %8.1f GB/s   col-major %8.1f GB/s   (cm/pp = %.2f)\n", cw, a, b, b/a);
  }
  return 0;
}
