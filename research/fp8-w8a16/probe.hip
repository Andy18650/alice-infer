// Standalone gfx90a E4M3FN -> FP16 MFMA research. No engine dependencies.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP(x) do { auto e = (x); if (e != hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while (0)
using H4 = _Float16 __attribute__((ext_vector_type(4)));
using U2 = uint32_t __attribute__((ext_vector_type(2)));
using F16 = float __attribute__((ext_vector_type(16)));

// Explicit op_sel_hi on shift: both halfwords must use the low-half shift=1.
// v_perm selectors 0..3 select src1 bytes, 4..7 select src0 bytes.
template<bool Rescale = true, bool Delay = true>
__device__ __forceinline__ H4 expand(uint32_t w, H4 previous = H4{}) {
  uint32_t lo, hi;
  const uint32_t mask = 0xbfffbfff, scale = 0x5c005c00;
  asm volatile(
      "v_perm_b32 %0, 0, %2, %3\n\t"
      "v_perm_b32 %1, 0, %2, %4\n\t"
      "v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]\n\t"
      "v_pk_ashrrev_i16 %1, 1, %1 op_sel_hi:[0,1]\n\t"
      "v_and_b32 %0, %5, %0\n\t"
      "v_and_b32 %1, %5, %1"
      : "=&v"(lo), "=&v"(hi)
      : "v"(w), "s"(0x01040004u), "s"(0x03040204u), "s"(mask), "v"(previous));
  if constexpr (Rescale) {
    asm volatile("v_pk_mul_f16 %0, %0, %2\n\tv_pk_mul_f16 %1, %1, %2"
                 : "+v"(lo), "+v"(hi) : "s"(scale));
    if constexpr (Delay) asm volatile("s_nop 3" : "+v"(lo), "+v"(hi));
  }
  return __builtin_bit_cast(H4, (U2{lo, hi}));
}

__global__ void conversion_test(const uint32_t* in, U2* scaled, U2* raw, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) {
    scaled[t] = __builtin_bit_cast(U2, expand(in[t]));
    raw[t] = __builtin_bit_cast(U2, expand<false>(in[t]));
  }
}

// One arbitrary dense 32x32x8 tile, output kept in calculator register order.
template<bool Rescale>
__global__ void tile_test(const H4* a, const uint32_t* b, float* out) {
  int lane = threadIdx.x;
  F16 c = {};
  c = __builtin_amdgcn_mfma_f32_32x32x8f16(a[lane], expand<Rescale>(b[lane]), c, 0, 0, 0);
  for (int r = 0; r < 16; ++r) out[lane*16+r] = c[r];
}

// Mode 0: pre-expanded register operand. Mode 1: register conversion each MFMA.
// Mode 2/3: eight staged global operands, FP16 / FP8 respectively. Activation
// stays in registers. Each wave computes an independent repeated tile GEMM.
template<int Mode>
__global__ __launch_bounds__(256) void throughput(
    const H4* a, const uint32_t* seed, const U2* half_seed,
    const uint32_t* stream8, const U2* stream16, float* out,
    uint64_t* stamps, int steps) {
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  int lane = threadIdx.x % 64;
  int wave = tid/64;
  H4 act = a[lane];
  uint32_t w = seed[lane];
  H4 b = __builtin_bit_cast(H4, half_seed[lane]);
  F16 c = {};
  uint64_t begin = clock64();
  uint64_t wall_begin = wall_clock64();
#pragma clang loop unroll(disable)
  for (int s = 0; s < steps; s += 8) {
    uint32_t q[8]; U2 h[8];
    if constexpr (Mode >= 2) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        size_t index = (size_t(wave)*steps+s+j)*64+lane;
        if constexpr (Mode == 3) q[j] = stream8[index];
        else h[j] = stream16[index];
      }
      // Prevent conversion/MFMA from sinking the independent loads into a
      // serialized load/wait/use sequence. This is a batching control, not
      // an assertion that load latency is hidden; inspect actual ISA.
      if constexpr (Mode == 3)
        asm volatile("" : : "v"(q[0]), "v"(q[1]), "v"(q[2]), "v"(q[3]), "v"(q[4]), "v"(q[5]), "v"(q[6]), "v"(q[7]));
      else
        asm volatile("" : : "v"(h[0]), "v"(h[1]), "v"(h[2]), "v"(h[3]), "v"(h[4]), "v"(h[5]), "v"(h[6]), "v"(h[7]));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      H4 weights, next;
      if constexpr (Mode == 0) weights = b;
      if constexpr (Mode == 1) {
        // Prepare NEXT operand before consuming CURRENT b. The intervening
        // MFMA and next conversion provide multiply->consumer separation,
        // without an explicit s_nop. Earlyclobber keeps both banks distinct.
        next = expand<true, false>(w, b);
        __builtin_amdgcn_sched_barrier(0);
        weights = b;
      }
      if constexpr (Mode == 2) weights = __builtin_bit_cast(H4, h[j]);
      if constexpr (Mode == 3) weights = expand(q[j], b);
      c = __builtin_amdgcn_mfma_f32_32x32x8f16(act, weights, c, 0, 0, 0);
      if constexpr (Mode == 1 || Mode == 3) {
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (Mode == 1) b = next;
        else b = weights;
      }
    }
  }
  uint64_t end = clock64();
  uint64_t wall_end = wall_clock64();
  if (lane == 0) {
    stamps[wave*4] = begin; stamps[wave*4+1] = end;
    stamps[wave*4+2] = wall_begin; stamps[wave*4+3] = wall_end;
  }
  for (int r = 0; r < 16; ++r) out[tid*16+r] = c[r];
}

// Rotate through three distinct matrices across K steps, so streamed operands
// cannot silently be replaced by a single invariant seed. No timed packing.
__global__ void fill_stream(const uint32_t* seeds, const U2* halves,
                            uint32_t* q, U2* h, size_t n) {
  size_t t = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (t < n) {
    int index = int((t/64)%3)*64+int(t%64);
    q[t] = seeds[index]; h[t] = halves[index];
  }
}

float decode(uint8_t x) {
  int e=(x>>3)&15, m=x&7;
  if (e==15 && m==7) return NAN;
  float v = e ? std::ldexp(float(8+m), e-10) : std::ldexp(float(m), -9);
  return x&128 ? -v : v;
}
uint16_t half_bits(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
uint16_t raw_bits(uint8_t x) { return uint16_t((uint16_t(x&128)<<8) | (uint16_t(x&127)<<7)); }
uint32_t pack(const uint8_t* x) { return uint32_t(x[0]) | uint32_t(x[1])<<8 | uint32_t(x[2])<<16 | uint32_t(x[3])<<24; }

template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP(hipMalloc(&p, n*sizeof(T))); }
  ~Device() { if(p) (void)hipFree(p); }
  void put(const std::vector<T>& x) { HIP(hipMemcpy(p,x.data(),x.size()*sizeof(T),hipMemcpyHostToDevice)); }
  std::vector<T> get(size_t n) const { std::vector<T> x(n); HIP(hipMemcpy(x.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return x; }
  Device(const Device&)=delete;
};
void require(bool ok, const char* what) { if(!ok) throw std::runtime_error(what); }

void check_conversion() {
  // Every byte value in every position, with independently varying neighbors.
  std::vector<uint32_t> words;
  for (int pos=0;pos<4;++pos) for(int x=0;x<256;++x) {
    uint8_t b[4]; for(int j=0;j<4;++j) b[j]=uint8_t(x*37+j*53+pos*11);
    b[pos]=uint8_t(x); words.push_back(pack(b));
  }
  Device<uint32_t> in(words.size()); Device<U2> scaled(words.size()), raw(words.size());
  in.put(words);
  conversion_test<<<4,256>>>(in.p,scaled.p,raw.p,int(words.size())); HIP(hipGetLastError());
  auto s=scaled.get(words.size()), r=raw.get(words.size());
  int finite=0;
  for(size_t t=0;t<words.size();++t) for(int j=0;j<4;++j) {
    uint8_t b=uint8_t(words[t]>>(j*8));
    uint16_t rb=uint16_t(r[t][j/2]>>(16*(j%2)));
    uint16_t sb=uint16_t(s[t][j/2]>>(16*(j%2)));
    require(rb==raw_bits(b), "raw conversion mismatch");
    if((b&127)!=127) { require(sb==half_bits(decode(b)), "rescaled conversion mismatch"); ++finite; }
  }
  std::printf("PASS conversion: 1024 mixed dwords, %d finite elements; raw bits also checked for NaN encodings\n",finite);
}

void check_tile() {
  std::vector<H4> a(64); std::vector<uint32_t> b(64);
  float host_a[32][8], host_b[8][32];
  for(int lane=0;lane<64;++lane) {
    uint8_t bytes[4];
    for(int t=0;t<4;++t) {
      int k=4*(lane/32)+t, col=lane%32;
      a[lane][t]=_Float16(((col*13+k*7)%17-8)*0.125f);
      host_a[col][k]=float(a[lane][t]);
      // All 254 finite encodings appear across the operand plus repeated zeros.
      int x=(lane*4+t)%254; bytes[t]=uint8_t(x<127?x:x+1);
      host_b[k][col]=decode(bytes[t]);
    }
    b[lane]=pack(bytes);
  }
  Device<H4> da(64); Device<uint32_t> db(64); Device<float> out(1024);
  da.put(a); db.put(b);
  tile_test<true><<<1,64>>>(da.p,db.p,out.p); HIP(hipGetLastError());
  auto y=out.get(1024);
  for(int lane=0;lane<64;++lane) for(int r=0;r<16;++r) {
    int row=8*(r/4)+4*(lane/32)+r%4, col=lane%32;
    double ref=0; for(int k=0;k<8;++k) ref+=double(host_a[row][k])*host_b[k][col];
    require(std::abs(y[lane*16+r]-ref)<=1e-5*(1+std::abs(ref)), "dense tile FP64 oracle mismatch");
  }
  // Isolate all seven positive/negative subnormal magnitudes, both operand halves.
  for(int x=1;x<=7;++x) for(int sign: {0,128}) {
    for(auto& v:a) v=H4{1,1,1,1};
    std::fill(b.begin(),b.end(),uint32_t(x+sign)*0x01010101u);
    da.put(a); db.put(b);
    tile_test<true><<<1,64>>>(da.p,db.p,out.p); HIP(hipGetLastError());
    y=out.get(1024); for(float v:y) require(v==8*decode(uint8_t(x+sign)), "MFMA restored subnormal mismatch");
    tile_test<false><<<1,64>>>(da.p,db.p,out.p); HIP(hipGetLastError());
    y=out.get(1024); for(float v:y) require(v==0, "expected raw MFMA input flush absent");
  }
  std::puts("PASS dense 32x32x8 FP64 oracle (all 1024 outputs); all 14 signed FP8 subnormals restored, raw path flushes");
}

int main(int argc,char** argv) try {
  int steps=argc>1?std::stoi(argv[1]):4096, rounds=argc>2?std::stoi(argv[2]):9;
  require(steps>0 && steps%8==0 && rounds>0,"steps must be positive multiple of 8; rounds positive");
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,0));
  std::printf("device=%s arch=%s CUs=%d steps=%d rounds=%d; no model/attention context\n",prop.name,prop.gcnArchName,prop.multiProcessorCount,steps,rounds);
  check_conversion(); check_tile();
  int blocks=prop.multiProcessorCount, threads=blocks*256, waves=threads/64;
  std::vector<H4> a(64); std::vector<uint32_t> q(192); std::vector<U2> h(192);
  double dot[3][32][32] = {};
  float av[32][8], bv[3][8][32];
  for(int lane=0;lane<64;++lane) for(int t=0;t<4;++t) {
    int k=4*(lane/32)+t,row=lane%32;
    a[lane][t]=_Float16(((row*3+k*5)%7-3)*0.125f); av[row][k]=float(a[lane][t]);
  }
  for(int m=0;m<3;++m) for(int lane=0;lane<64;++lane) {
    uint8_t bytes[4]; H4 v;
    for(int t=0;t<4;++t) {
      int x=(lane*19+t*7+m*13)%57; // zero/subnormal/normal, mixed signs
      bytes[t]=uint8_t(x | (((lane+t+m)%3==0)?128:0));
      v[t]=_Float16(decode(bytes[t])); bv[m][4*(lane/32)+t][lane%32]=float(v[t]);
    }
    q[m*64+lane]=pack(bytes); h[m*64+lane]=std::bit_cast<U2>(v);
  }
  for(int m=0;m<3;++m) for(int i=0;i<32;++i) for(int j=0;j<32;++j)
    for(int k=0;k<8;++k) dot[m][i][j]+=double(av[i][k])*bv[m][k][j];
  Device<H4> da(64); Device<uint32_t> dq(192); Device<U2> dh(192);
  Device<float> out(size_t(threads)*16); Device<uint64_t> stamps(waves*4);
  size_t records=size_t(waves)*steps*64;
  Device<uint32_t> stream8(records); Device<U2> stream16(records);
  da.put(a); dq.put(q); dh.put(h);
  fill_stream<<<(records+255)/256,256>>>(dq.p,dh.p,stream8.p,stream16.p,records); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
  const char* names[]={"register-fp16","register-fp8-convert","stream-fp16-batch8","stream-fp8-batch8"};
  auto launch=[&](int mode) {
#define LAUNCH(M) throughput<M><<<blocks,256>>>(da.p,dq.p,dh.p,stream8.p,stream16.p,out.p,stamps.p,steps)
    switch(mode) {case 0: LAUNCH(0); break; case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; case 3: LAUNCH(3); break;}
#undef LAUNCH
    HIP(hipGetLastError());
  };
  for(int m=0;m<4;++m) { launch(m); HIP(hipDeviceSynchronize()); }
  hipEvent_t start, stop; HIP(hipEventCreate(&start)); HIP(hipEventCreate(&stop));
  std::vector<double> rates[4];
  for(int round=0;round<rounds;++round) for(int order=0;order<4;++order) {
    int mode=(order+round)%4;
    HIP(hipEventRecord(start)); for(int repeat=0;repeat<8;++repeat) launch(mode);
    HIP(hipEventRecord(stop)); HIP(hipEventSynchronize(stop));
    float ms; HIP(hipEventElapsedTime(&ms,start,stop)); ms/=8;
    auto y=out.get(size_t(threads)*16); auto st=stamps.get(waves*4);
    double cycles=0, ticks=0, maxerr=0;
    for(int wave=0;wave<waves;++wave) { cycles+=st[wave*4+1]-st[wave*4]; ticks+=st[wave*4+3]-st[wave*4+2]; }
    for(int tid=0;tid<threads;++tid) for(int r=0;r<16;++r) {
      int lane=tid%64,row=8*(r/4)+4*(lane/32)+r%4,col=lane%32;
      double ref=dot[0][row][col]*steps;
      if(mode>=2) { ref=0; int wave=tid/64; for(int m=0;m<3;++m) {
        // fill_stream's phase includes each wave's base offset.
        int first=(m-(wave*steps)%3+3)%3;
        int count=first<steps?1+(steps-1-first)/3:0;
        ref+=count*dot[m][row][col];
      }}
      double err=std::abs(y[tid*16+r]-ref); maxerr=std::max(maxerr,err);
      if (!(std::isfinite(y[tid*16+r]) && err<=0.001*(1+std::abs(ref)))) {
        std::fprintf(stderr,"mismatch mode=%d tid=%d r=%d got=%.9g ref=%.12g err=%.9g\n",mode,tid,r,y[tid*16+r],ref,err);
        throw std::runtime_error("throughput all-output oracle mismatch");
      }
    }
    double tf=double(waves)*steps*16384/(ms*1e9);
    rates[mode].push_back(tf);
    std::printf("round=%d mode=%s ms=%.6f TFLOPs=%.3f cycles_per_mfma=%.3f core_MHz=%.1f max_abs_error=%.6g\n",round,names[mode],ms,tf,cycles/(waves*double(steps)),cycles/ticks*double(wall_rate)/1000,maxerr);
  }
  for(int mode=0;mode<4;++mode) { auto v=rates[mode]; std::sort(v.begin(),v.end()); std::printf("SUMMARY %s median_TFLOPs=%.3f\n",names[mode],v[v.size()/2]); }
  HIP(hipEventDestroy(start)); HIP(hipEventDestroy(stop));
  std::puts("PASS all throughput outputs against independent FP64 reference");
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
