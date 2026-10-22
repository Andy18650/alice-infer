#pragma once
#include <cstdint>
#include <vector>
inline uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
inline int unit(bool b,int row,int col,uint32_t seed){return int(mix(uint32_t(row+1)*0x9e3779b9u^uint32_t(col+1)*0x85ebca6bu^seed^(b?0xc8013ea4u:0xa341316cu))%9)-4;}
inline uint16_t half_bits(int x){int m=x<0?-x:x;uint16_t v=m==0?0:uint16_t(m<=2?0x2800+(m<<10):0x2c00+(m<<9));return v|(x<0?0x8000:0);}
struct Data{
  int m,n,k;std::vector<int8_t> a,b;
  Data(int M,int N,int K,uint32_t sa,uint32_t sb):m(M),n(N),k(K),a(size_t(M)*K),b(size_t(K)*N){
    for(int i=0;i<M;++i)for(int q=0;q<K;++q)a[size_t(i)*K+q]=int8_t(unit(false,i,q,sa));
    for(int q=0;q<K;++q)for(int j=0;j<N;++j)b[size_t(q)*N+j]=int8_t(unit(true,q,j,sb));
  }
  // [K32 chunk][half-CTA row/column panel][qK8][pair][lane][4 dwords].
  // Two32x8 MFMA fragments are interleaved per lane for one b128 read.
  std::vector<uint32_t> pack(bool is_b,int tile)const{
    int extent=is_b?n:m,padded=(extent+tile-1)/tile*tile,half=tile/2,panels=padded/half,kt=(k+31)/32,pairs=tile/128;
    size_t panel_words=size_t(half)*32/2;std::vector<uint32_t> out(size_t(kt)*panels*panel_words);
    for(int s=0;s<kt;++s)for(int p=0;p<panels;++p)for(int q=0;q<4;++q)for(int pair=0;pair<pairs;++pair)for(int lane=0;lane<64;++lane)for(int w=0;w<4;++w){
      int pos=p*half+(pair*2+w/2)*32+lane%32,kk=s*32+q*8+(lane/32)*4+(w%2)*2;uint32_t bits=0;
      for(int h=0;h<2;++h)if(pos<extent&&kk+h<k){int v=is_b?b[size_t(kk+h)*n+pos]:a[size_t(pos)*k+kk+h];bits|=uint32_t(half_bits(v))<<(16*h);}
      size_t off=(size_t(s)*panels+p)*panel_words+size_t(q*pairs+pair)*256+lane*4+w;out[off]=bits;
    }
    return out;
  }
  float ref(int i,int j)const{int64_t dot=0;for(int q=0;q<k;++q)dot+=int(a[size_t(i)*k+q])*int(b[size_t(q)*n+j]);return float(double(dot)/256.0);}
};
