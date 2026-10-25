#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
// Research-only host checking/timing helpers; no library GEMM reference.
#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../native-gemm/data_scaled.hpp"

#define HIP(expr) do { auto status=(expr); if(status!=hipSuccess) \
    throw std::runtime_error(std::string(#expr)+": "+hipGetErrorString(status)); } while(0)

template<class T> struct Buffer {
    T* data=nullptr;
    explicit Buffer(size_t count) { HIP(hipMalloc(&data,std::max(size_t(1),count)*sizeof(T))); }
    ~Buffer() { if(data) (void)hipFree(data); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
struct Event {
    hipEvent_t event;
    Event() { HIP(hipEventCreate(&event)); }
    ~Event() { (void)hipEventDestroy(event); }
};
struct Reference {
    bool full;
    std::vector<size_t> indices;
    std::vector<float> values;
};
inline Reference make_reference(const ScaledMatrices& matrices,bool first_row=false) {
    const int m=first_row?1:matrices.rows,n=matrices.columns,k=matrices.reduction;
    Reference r;
    r.full=uint64_t(m)*n*std::max(k,1)<=30000000;
    if(r.full) for(size_t i=0;i<size_t(m)*n;++i) r.indices.push_back(i);
    else {
        std::mt19937 random(314159);
        for(int i=0;i<2048;++i) r.indices.push_back(size_t(random())%(size_t(m)*n));
        for(int row=0;row<m;row+=32) for(int col:{0,n-1}) r.indices.push_back(size_t(row)*n+col);
        for(int col=0;col<n;col+=32) for(int row:{0,m-1}) r.indices.push_back(size_t(row)*n+col);
        for(int row:{0,m-1}) for(int col:{0,n-1}) r.indices.push_back(size_t(row)*n+col);
        std::sort(r.indices.begin(),r.indices.end());
        r.indices.erase(std::unique(r.indices.begin(),r.indices.end()),r.indices.end());
    }
    for(size_t index:r.indices) r.values.push_back(matrices.reference_dot(index/n,index%n));
    return r;
}
inline void verify(const std::vector<float>& c,size_t count,const Reference& r) {
    for(size_t i=0;i<count;++i) if(!std::isfinite(c[i+16])) throw std::runtime_error("nonfinite/unwritten C");
    for(size_t i=0;i<r.indices.size();++i) if(c[16+r.indices[i]]!=r.values[i]) {
        std::fprintf(stderr,"mismatch %zu expected %.9g got %.9g\n",r.indices[i],r.values[i],c[16+r.indices[i]]);
        throw std::runtime_error("logical reference mismatch");
    }
    for(size_t i=0;i<16;++i)
        if(std::bit_cast<uint32_t>(c[i])!=0xffffffffu||std::bit_cast<uint32_t>(c[count+16+i])!=0xffffffffu)
            throw std::runtime_error("C guard modified");
}
struct Clocks { double body_us,mhz; };
inline Clocks check_clocks(const std::vector<uint64_t>& stamps,int blocks,int waves,int rate,double event_us) {
    uint64_t first=UINT64_MAX,last=0;
    std::vector<double> ratios;
    for(int block=0;block<blocks;++block) {
        uint64_t cb=UINT64_MAX,ce=0,sb=UINT64_MAX,se=0;
        for(int w=0;w<waves;++w) {
            const auto* s=&stamps[(size_t(block)*waves+w)*4];
            if(s[1]<=s[0]||s[3]<=s[2]) throw std::runtime_error("nonmonotonic clocks");
            cb=std::min(cb,s[0]);ce=std::max(ce,s[1]);sb=std::min(sb,s[2]);se=std::max(se,s[3]);
        }
        first=std::min(first,sb);last=std::max(last,se);
        const double ratio=double(ce-cb)/(double(se-sb)*1000/rate);
        if(!std::isfinite(ratio)||ratio<50||ratio>3000) throw std::runtime_error("invalid core/steady ratio");
        ratios.push_back(ratio);
    }
    const double body=double(last-first)*1000/rate;
    if(body>event_us*1.05+10) throw std::runtime_error("body span outside HIP event");
    std::sort(ratios.begin(),ratios.end());
    return {body,(ratios[(ratios.size()-1)/2]+ratios[ratios.size()/2])/2};
}

// CPU packing deliberately uses a different decomposition from b_index().
inline size_t cpu_b_index(int k,int n,int storage_n) {
#if COMMON_B_FORMAT == 0
    const int tile=n/128,stage=k/32,q=(k-stage*32)/8;
    const int col=n-tile*128;
    const int half_k=(k%8)/4;
    return (size_t(stage)*(storage_n/128)+tile)*4096+q*1024
        +(col%32+32*half_k)*16+(col/32)*4+k%4;
#elif COMMON_B_FORMAT == 1
    return size_t(k)*storage_n+n;
#else
    constexpr int tile_n=COMMON_B_FORMAT==2?128:256;
    const size_t group=(k/128)*(storage_n/tile_n)+n/tile_n;
    return group*128*tile_n+(k%128)*tile_n+n%tile_n;
#endif
}
