#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

// Input construction is shared host/device; reference addressing and matrix
// multiplication below are independent of the generated pipeline instructions.
__host__ __device__ inline uint32_t input_mix(uint32_t x) {
  x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
__host__ __device__ inline uint16_t unit_half(int x) {
  const int m=x<0?-x:x;
  const uint16_t bits=m==0?0:uint16_t(m<=2?0x2800+(m<<10):0x2c00+(m<<9));
  return uint16_t(bits|(x<0?0x8000:0)); // x / 16, -4..4.
}
__host__ __device__ inline uint32_t operand_word(uint32_t index,int pattern,int loads,int stride,uint32_t seed_a,uint32_t seed_b) {
  if(!pattern) return 0x30003000u|((index*1664525u+1013904223u)&0x80008000u);
  const uint32_t local=index%uint32_t(256*stride);
  const bool is_b=loads==1?(local%uint32_t(4*stride)>=2):(local>=uint32_t(128*stride));
  const uint32_t seed=is_b?seed_b:seed_a;
  const int lo=int(input_mix(index^input_mix(seed+0xa341316cu))%9)-4;
  const int hi=int(input_mix(index^input_mix(seed+0xc8013ea4u))%9)-4;
  return uint32_t(unit_half(lo))|(uint32_t(unit_half(hi))<<16);
}
inline int half_units(uint16_t h) {
  int x;
  switch(h&0x7fff) {
    case 0:x=0;break;case 0x2c00:x=1;break;case 0x3000:x=2;break;
    case 0x3200:x=3;break;case 0x3400:x=4;break;
    default:throw std::runtime_error("reference input is not one of the constructed FP16 values");
  }
  return h&0x8000?-x:x;
}
inline size_t reference_word(int wave,int record,int lane,int logical_word,int records,int loads,int stride) {
  const size_t base=(size_t(wave)*records+record)*size_t(256*stride);
  if(loads==1) return base+size_t(lane)*4*stride+logical_word;
  if(loads==2) return base+size_t(lane)*2*stride+size_t(logical_word/2)*128*stride+logical_word%2;
  return base+size_t(logical_word)*64*stride+size_t(lane)*stride;
}
inline std::vector<float> matrix_reference(const std::vector<uint32_t>& input,int mode,int loads,int stride,int steps,int records) {
  // Every product is an integer / 256. At <=8192 steps every intermediate
  // integer magnitude is <= 8*8192*16 = 2^20, so all FP32 sums are exact.
  if(steps>8192) throw std::runtime_error("exact verification currently supports steps <=8192; use --verify 0 for longer performance runs");
  std::vector<int> counts(records);
  for(int n=0;n<steps;++n) ++counts[(mode==0||mode==3)?0:mode==2?n:n%records];
  std::vector<int64_t> totals(4*1024);
  auto value=[&](int wave,int rec,int lane,int word,int half) {
    const uint32_t bits=input.at(reference_word(wave,rec,lane,word,records,loads,stride));
    return half_units(uint16_t(bits>>(16*half)));
  };
  for(int wave=0;wave<4;++wave) for(int rec=0;rec<records;++rec) if(counts[rec]) {
    int a[32][8],b[8][32];
    for(int row=0;row<32;++row) for(int k=0;k<8;++k)
      a[row][k]=value(wave,rec,row+32*(k/4),(k%4)/2,k%2);
    for(int col=0;col<32;++col) for(int k=0;k<8;++k)
      b[k][col]=value(wave,rec,col+32*(k/4),2+(k%4)/2,k%2);
    for(int row=0;row<32;++row) for(int col=0;col<32;++col) {
      int dot=0;
      for(int k=0;k<8;++k) dot+=a[row][k]*b[k][col];
      totals[wave*1024+row*32+col]+=int64_t(counts[rec])*dot;
    }
  }
  std::vector<float> expected(4096);
  // Documented 32x32x8 MFMA result mapping, not the output-store order
  // interpreted as a row-major matrix.
  for(int wave=0;wave<4;++wave) for(int lane=0;lane<64;++lane) for(int r=0;r<16;++r) {
    const int row=8*(r/4)+4*(lane/32)+r%4,col=lane%32;
    expected[(wave*64+lane)*16+r]=float(double(totals[wave*1024+row*32+col])/256.0);
  }
  return expected;
}
struct Verification { int mismatches=0,guard_errors=0; double max_error=0; };
inline Verification verify_matrix(const std::vector<float>& output,const std::vector<float>& expected,bool detail) {
  Verification result;
  for(int i=0;i<16;++i) {
    result.guard_errors+=std::bit_cast<uint32_t>(output[i])!=0xffffffffu;
    result.guard_errors+=std::bit_cast<uint32_t>(output[16+4096+i])!=0xffffffffu;
  }
  for(int i=0;i<4096;++i) {
    const float actual=output[i+16];
    const double error=std::isfinite(actual)?std::abs(double(actual)-expected[i]):INFINITY;
    if(error!=0) {
      if(detail && result.mismatches<4) std::fprintf(stderr,"MISMATCH wave=%d lane=%d acc=%d expected=%.9g actual=%.9g bits=%08x\n",i/1024,(i/16)%64,i%16,expected[i],actual,std::bit_cast<uint32_t>(actual));
      ++result.mismatches;
    }
    result.max_error=std::max(result.max_error,error);
  }
  return result;
}
