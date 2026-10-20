#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
__host__ __device__ inline uint32_t mix(uint32_t x) { x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16); }
__host__ __device__ inline int value(bool b,int row,int col,uint32_t seed) {
  return int(mix(uint32_t(row+1)*0x9e3779b9u^uint32_t(col+1)*0x85ebca6bu^seed^(b?0xc8013ea4u:0xa341316cu))%9)-4;
}
__host__ __device__ inline uint16_t bits(int x) {
  int m=x<0?-x:x;uint16_t v=m==0?0:uint16_t(m<=2?0x2800+(m<<10):0x2c00+(m<<9));return v|(x<0?0x8000:0);
}
// GPU procedural LDS initializer: packed record -> logical A/B coordinates.
__host__ __device__ inline uint32_t local_word(int i,uint32_t sa,uint32_t sb) {
  int f=i/512,block=i%512/128,lane=i%128/2,w=i%2;
  int k=f*8+(lane/32)*4+w*2,pos=(block%2)*32+lane%32;
  bool b=block>=2;
  return uint32_t(bits(value(b,b?k:pos,b?pos:k,b?sb:sa)))|
         (uint32_t(bits(value(b,b?k+1:pos,b?pos:k+1,b?sb:sa)))<<16);
}
struct Data {
  int k;std::vector<int8_t> au,bu;std::vector<uint16_t> a,b;
  Data(int K,uint32_t sa,uint32_t sb):k(K),au(size_t(64)*K),bu(size_t(K)*64),a(au.size()),b(bu.size()) {
    for(int i=0;i<64;++i) for(int q=0;q<K;++q) { int x=value(false,i,q,sa);au[size_t(i)*K+q]=int8_t(x);a[size_t(i)*K+q]=bits(x); }
    for(int q=0;q<K;++q) for(int j=0;j<64;++j) { int x=value(true,q,j,sb);bu[size_t(q)*64+j]=int8_t(x);b[size_t(q)*64+j]=bits(x); }
  }
  std::vector<uint32_t> pack(int padded_k) const {
    std::vector<uint32_t> out(size_t(padded_k/8)*512);
    // CPU mapping from row-major FP16 inputs, independent of GPU initializer.
    for(int f=0;f<padded_k/8;++f) for(int tile=0;tile<2;++tile) for(int lane=0;lane<64;++lane) for(int w=0;w<2;++w) {
      int q=f*8+4*(lane/32)+2*w,pos=tile*32+lane%32;
      uint32_t av=0,bv=0;
      for(int h=0;h<2;++h) if(q+h<k) {
        av|=uint32_t(a[size_t(pos)*k+q+h])<<(16*h);
        bv|=uint32_t(b[size_t(q+h)*64+pos])<<(16*h);
      }
      out[size_t(f)*512+tile*128+lane*2+w]=av;
      out[size_t(f)*512+256+tile*128+lane*2+w]=bv;
    }
    return out;
  }
  std::vector<float> reference(int repetitions=1,int records=0) const {
    std::vector<int> counts;
    if(records) { counts.resize(records);for(int n=0;n<repetitions;++n) ++counts[n%records]; }
    std::vector<float> out(4096);
    for(int i=0;i<64;++i) for(int j=0;j<64;++j) {
      int64_t dot=0;
      for(int q=0;q<k;++q) dot+=int(au[size_t(i)*k+q])*int(bu[size_t(q)*64+j])*(records?counts[q/8]:1);
      out[i*64+j]=float(double(dot)/256.0);
    }
    return out;
  }
};
struct Check { int bad=0,guards=0;double maxerr=0; };
inline Check check_output(const std::vector<float>& y,const std::vector<float>& ref,bool detail=true) {
  Check c;
  for(int i=0;i<16;++i) { c.guards+=std::bit_cast<uint32_t>(y[i])!=0xffffffffu;c.guards+=std::bit_cast<uint32_t>(y[4112+i])!=0xffffffffu; }
  for(int i=0;i<4096;++i) {
    double e=std::isfinite(y[i+16])?std::abs(double(y[i+16])-ref[i]):INFINITY;
    if(e) { if(detail&&c.bad<4)std::fprintf(stderr,"MISMATCH row=%d col=%d expected=%.9g actual=%.9g\n",i/64,i%64,ref[i],y[i+16]);++c.bad; }
    c.maxerr=std::max(c.maxerr,e);
  }
  return c;
}
