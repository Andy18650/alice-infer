#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// GPU address decoder, independent of the CPU packer's loop nest. B is always
// one resident allocation used by BOTH GEMM and GEMV without an intervening pack.
__device__ __forceinline__ size_t b_index(int k,int n,int storage_n) {
#if COMMON_B_FORMAT == 0
    const size_t panel=size_t(k>>5)*(storage_n>>7)+(n>>7);
    const int lane=(n&31)+((k&7)>>2)*32;
    return panel*4096+((k&31)>>3)*1024+lane*16+((n&127)>>5)*4+(k&3);
#elif COMMON_B_FORMAT == 1
    return size_t(k)*storage_n+n;
#else
    constexpr int bn=COMMON_B_FORMAT==2?128:256;
    return (size_t(k>>7)*(storage_n/bn)+n/bn)*(128*bn)+(k&127)*bn+n%bn;
#endif
}

__device__ __forceinline__ uint32_t byte_perm(uint32_t a,uint32_t b,uint32_t selector) {
    uint32_t result;
    asm("v_perm_b32 %0, %1, %2, %3" : "=v"(result) : "v"(a),"v"(b),"s"(selector));
    return result;
}

__device__ __forceinline__ uint4 transpose_bytes(uint4 input) {
    const uint32_t a=byte_perm(input.y,input.x,0x05010400u);
    const uint32_t b=byte_perm(input.w,input.z,0x05010400u);
    const uint32_t c=byte_perm(input.y,input.x,0x07030602u);
    const uint32_t d=byte_perm(input.w,input.z,0x07030602u);
    return {byte_perm(b,a,0x05040100u),byte_perm(b,a,0x07060302u),
            byte_perm(d,c,0x05040100u),byte_perm(d,c,0x07060302u)};
}

__device__ __forceinline__ uint2 expand_word(uint32_t input) {
    uint2 out;
    asm("v_perm_b32 %0, 0, %2, %3\n\t"
        "v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]\n\t"
        "v_and_b32 %0, %5, %0\n\t"
        "v_pk_mul_f16 %0, %0, %6\n\t"
        "v_perm_b32 %1, 0, %2, %4\n\t"
        "v_pk_ashrrev_i16 %1, 1, %1 op_sel_hi:[0,1]\n\t"
        "v_and_b32 %1, %5, %1\n\t"
        "v_pk_mul_f16 %1, %1, %6"
        : "=&v"(out.x),"=&v"(out.y) : "v"(input),"s"(0x01040004u),"s"(0x03040204u),
          "s"(0xbfffbfffu),"s"(0x5c005c00u));
    return out;
}

__device__ __forceinline__ uint4 load_four_rows(const uint8_t* b,int k,int n,int storage_n) {
#if COMMON_B_FORMAT == 0
    // Four column records, each containing four K bytes. Transpose to the
    // same K-row/N4 representation consumed by all other decode variants.
    const auto* p=reinterpret_cast<const uint32_t*>(b+b_index(k,n,storage_n));
    return transpose_bytes({p[0],p[4],p[8],p[12]});
#else
    return {*reinterpret_cast<const uint32_t*>(b+b_index(k,n,storage_n)),
            *reinterpret_cast<const uint32_t*>(b+b_index(k+1,n,storage_n)),
            *reinterpret_cast<const uint32_t*>(b+b_index(k+2,n,storage_n)),
            *reinterpret_cast<const uint32_t*>(b+b_index(k+3,n,storage_n))};
#endif
}

// Column-coalesced, K-split SIMT strategy derived from the existing CM GEMV.
// One lane owns N4, four independent K streams, one K128 raw partial/group.
// Combine applies scales in ascending group order, matching the GEMM contract.
__global__ __launch_bounds__(64) void decode_partial(
    const uint8_t* __restrict__ b,const uint16_t* __restrict__ x,
    float* __restrict__ partial,uint64_t* stamps,int np,int k,int storage_n) {
    const int quads=np/4;
    const int groups=max(1,(k+127)/128);
    const int index=blockIdx.x*64+threadIdx.x;
    if(index>=quads*groups) return;
    const int g=index/quads,n=(index%quads)*4;
    uint64_t cb,sb;
    asm volatile("s_memtime %0\n\ts_memrealtime %1\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(cb),"=s"(sb) :: "memory");
    float sums[4][4]={};
    const int end=min(k,(g+1)*128);
    for(int inner=g*128;inner<end;inner+=4) {
        const uint4 words=load_four_rows(b,inner,n,storage_n);
        const uint32_t rows[4]={words.x,words.y,words.z,words.w};
#pragma unroll
        for(int j=0;j<4;++j) {
            const uint2 halves=expand_word(rows[j]);
            const float a=__half2float(__ushort_as_half(x[inner+j]));
            const uint16_t bits[4]={uint16_t(halves.x),uint16_t(halves.x>>16),uint16_t(halves.y),uint16_t(halves.y>>16)};
#pragma unroll
            for(int c=0;c<4;++c) sums[j][c]=fmaf(__half2float(__ushort_as_half(bits[c])),a,sums[j][c]);
        }
    }
#pragma unroll
    for(int c=0;c<4;++c) partial[size_t(g)*np+n+c]=(sums[0][c]+sums[1][c])+(sums[2][c]+sums[3][c]);
    uint64_t ce,se;
    asm volatile("s_memtime %0\n\ts_memrealtime %1\n\ts_waitcnt lgkmcnt(0)"
                 : "=s"(ce),"=s"(se) :: "memory");
    if(threadIdx.x==0) {
        stamps[size_t(blockIdx.x)*4]=cb;stamps[size_t(blockIdx.x)*4+1]=ce;
        stamps[size_t(blockIdx.x)*4+2]=sb;stamps[size_t(blockIdx.x)*4+3]=se;
    }
}

__global__ void decode_combine(const float* __restrict__ partial,const float* __restrict__ scales,
                                float* out,int n,int np,int groups) {
    const int col=blockIdx.x*256+threadIdx.x;
    if(col>=n) return;
    float sum=0;
    for(int g=0;g<groups;++g) sum=fmaf(partial[size_t(g)*np+col],scales[size_t(g)*(np/128)+col/128],sum);
    out[col]=sum;
}

__global__ void unpack_b(const uint8_t* input,uint8_t* output,int storage_n,int storage_k) {
    const size_t index=size_t(blockIdx.x)*256+threadIdx.x;
    if(index<size_t(storage_n)*storage_k) {
        const int k=index/storage_n,n=index%storage_n;
        output[index]=input[b_index(k,n,storage_n)];
    }
}

__global__ void primitive_check(const uint32_t* words,uint32_t* halves,uint4* transposed,int count) {
    const int i=blockIdx.x*256+threadIdx.x;
    if(i>=count) return;
    const uint2 h=expand_word(words[i]);halves[2*i]=h.x;halves[2*i+1]=h.y;
    transposed[i]=transpose_bytes({words[i],words[(i+1)%count],words[(i+2)%count],words[(i+3)%count]});
}
