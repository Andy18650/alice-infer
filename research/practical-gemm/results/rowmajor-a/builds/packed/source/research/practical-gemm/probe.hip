// Row-major FP16 activations vs the frozen balanced packed-A/FP8-B algorithm.
// All device work is native HIP; no BLAS/CK, including the reference.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../native-gemm/data_scaled.hpp"
#include "compute.hpp"
#ifdef STUDY_ROW_MAJOR_A
#include "rowmajor_kernel.hpp"
constexpr bool row_major_a = true;
#else
#include "../native-gemm/scaled_kernel.hpp"
constexpr bool row_major_a = false;
#endif

#define HIP(expr) do { auto status=(expr); if(status!=hipSuccess) \
    throw std::runtime_error(std::string(#expr)+": "+hipGetErrorString(status)); } while(0)

template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { HIP(hipMalloc(&data, std::max(size_t(1),count)*sizeof(T))); }
    ~Buffer() { if(data) (void)hipFree(data); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
struct Event {
    hipEvent_t event;
    Event() { HIP(hipEventCreate(&event)); }
    ~Event() { (void)hipEventDestroy(event); }
};

// Deliberately plain GPU packing control, not an optimized packer. It writes
// every padded destination word once and never reads nonexistent A rows.
__global__ void pack_activation(const uint16_t* source, uint32_t* destination,
                                int m, int k, int row_tiles, size_t words) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=words) return;
    const int word=index%4;
    const int lane=(index/4)%64;
    const int half=(index/256)%2;
    const int q=(index/512)%4;
    const int panel=(index/2048)%row_tiles;
    const int stage=index/(size_t(2048)*row_tiles);
    const int row=panel*128+half*64+(word/2)*32+lane%32;
    const int inner=stage*32+q*8+(lane/32)*4+(word%2)*2;
    uint32_t value=0;
    if(row<m) value=uint32_t(source[size_t(row)*k+inner])
                     | (uint32_t(source[size_t(row)*k+inner+1])<<16);
    destination[index]=value;
}

struct Options {
    int m=128,n=4096,k=4096,rounds=9,warmup_ms=200,scale=2,finite=0;
    uint32_t seed_a=1729,seed_b=2718;
    std::string prefix;
};
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(++i==argc) throw std::runtime_error("missing option value");
        const std::string value=argv[i];
        if(key=="--m") o.m=std::stoi(value);
        else if(key=="--n") o.n=std::stoi(value);
        else if(key=="--k") o.k=std::stoi(value);
        else if(key=="--rounds") o.rounds=std::stoi(value);
        else if(key=="--warmup-ms") o.warmup_ms=std::stoi(value);
        else if(key=="--scale-pattern") o.scale=std::stoi(value);
        else if(key=="--b-pattern") o.finite=std::stoi(value);
        else if(key=="--seed-a") o.seed_a=uint32_t(std::stoul(value));
        else if(key=="--seed-b") o.seed_b=uint32_t(std::stoul(value));
        else if(key=="--stamps-prefix") o.prefix=value;
        else throw std::runtime_error("unknown option "+key);
    }
    if(o.m<1||o.m>16384||o.n<1||o.n>32768||o.k<0||o.k>65536||o.k%32
       ||o.rounds<1||o.warmup_ms<0||o.scale<0||o.scale>3||o.finite<0||o.finite>1)
        throw std::runtime_error("initial study requires K multiple-of32 or zero; dimensions/options out of range");
    return o;
}

struct Reference {
    bool full;
    std::vector<size_t> indices;
    std::vector<float> values;
};
Reference reference(const ScaledMatrices& matrices) {
    const int m=matrices.rows,n=matrices.columns,k=matrices.reduction;
    Reference r;
    r.full=uint64_t(m)*n*std::max(k,1)<=30000000;
    if(r.full) {
        for(size_t i=0;i<size_t(m)*n;++i) r.indices.push_back(i);
    } else {
        std::mt19937 random(314159);
        for(int i=0;i<2048;++i) r.indices.push_back(size_t(random())%(size_t(m)*n));
        for(int row=0;row<m;row+=32)
            for(int col:{0,n-1}) r.indices.push_back(size_t(row)*n+col);
        for(int col=0;col<n;col+=32)
            for(int row:{0,m-1}) r.indices.push_back(size_t(row)*n+col);
        for(int row:{0,m-1}) for(int col:{0,n-1}) r.indices.push_back(size_t(row)*n+col);
        std::sort(r.indices.begin(),r.indices.end());
        r.indices.erase(std::unique(r.indices.begin(),r.indices.end()),r.indices.end());
    }
    for(size_t index:r.indices) r.values.push_back(matrices.reference_dot(index/n,index%n));
    return r;
}

void verify(const std::vector<float>& c,size_t count,const Reference& r) {
    for(size_t i=0;i<count;++i)
        if(!std::isfinite(c[i+16])) throw std::runtime_error("unwritten/nonfinite C");
    for(size_t i=0;i<r.indices.size();++i)
        if(c[16+r.indices[i]]!=r.values[i]) {
            std::fprintf(stderr,"mismatch %zu expected %.9g got %.9g\n",r.indices[i],r.values[i],c[16+r.indices[i]]);
            throw std::runtime_error("logical reference mismatch");
        }
    for(size_t i=0;i<16;++i)
        if(std::bit_cast<uint32_t>(c[i])!=0xffffffffu
           ||std::bit_cast<uint32_t>(c[count+16+i])!=0xffffffffu)
            throw std::runtime_error("C guard modified");
}

struct Clocks { double body_us,mhz; };
Clocks clocks(const std::vector<uint64_t>& stamps,int blocks,int rate,double event_us) {
    uint64_t first=UINT64_MAX,last=0;
    std::vector<double> ratios;
    for(int b=0;b<blocks;++b) {
        uint64_t cb=UINT64_MAX,ce=0,sb=UINT64_MAX,se=0;
        for(int w=0;w<4;++w) {
            const auto* s=&stamps[(size_t(b)*4+w)*4];
            if(s[1]<=s[0]||s[3]<=s[2]) throw std::runtime_error("nonmonotonic clocks");
            cb=std::min(cb,s[0]);ce=std::max(ce,s[1]);
            sb=std::min(sb,s[2]);se=std::max(se,s[3]);
        }
        first=std::min(first,sb);last=std::max(last,se);
        const double ratio=double(ce-cb)/(double(se-sb)*1000/rate);
        if(!std::isfinite(ratio)||ratio<50||ratio>3000) throw std::runtime_error("invalid core/steady ratio");
        ratios.push_back(ratio);
    }
    const double body=double(last-first)*1000/rate;
    if(body>event_us*1.05+10) throw std::runtime_error("body span outside independent HIP event");
    std::sort(ratios.begin(),ratios.end());
    return {body,(ratios[(ratios.size()-1)/2]+ratios[ratios.size()/2])/2};
}

int main(int argc,char** argv) try {
    const Options o=parse(argc,argv);
    HIP(hipSetDevice(0));
    hipDeviceProp_t device; HIP(hipGetDeviceProperties(&device,0));
    char bdf[64]; HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
    int rate; HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));
    if(std::string(bdf)!="0000:31:00.0"||device.multiProcessorCount!=104)
        throw std::runtime_error("study requires HIP0/BDF0000:31:00.0 and104CUs");
    const int mp=(o.m+127)/128*128,np=(o.n+127)/128*128,stages=o.k/32;
    const int blocks=(mp/128)*(np/128);
    int occupancy; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,native_gemm,256,16384));
    ScaledMatrices matrices(o.m,o.n,o.k,o.seed_a,o.seed_b);
    matrices.set_finite_pattern(o.finite!=0);matrices.set_scales(o.scale,true);
    std::vector<uint16_t> a(size_t(o.m)*o.k+32,0x7e00);
    for(size_t i=0;i<matrices.row_major_a_units.size();++i) a[i+16]=half_bits_from_unit(matrices.row_major_a_units[i]);
    const auto b=matrices.pack_operand(true,128,32);
    const size_t a_words=size_t(stages)*(mp/128)*2048;
    Buffer<uint16_t> da(a.size()); Buffer<uint32_t> db(b.size()),pa(a_words+32);
    Buffer<float> ds(matrices.scales.size()),dc(size_t(o.m)*o.n+32);
    Buffer<uint64_t> dt(size_t(blocks)*16);
    HIP(hipMemcpy(da.data,a.data(),a.size()*2,hipMemcpyHostToDevice));
    if(!b.empty()) HIP(hipMemcpy(db.data,b.data(),b.size()*4,hipMemcpyHostToDevice));
    if(!matrices.scales.empty()) HIP(hipMemcpy(ds.data,matrices.scales.data(),matrices.scales.size()*4,hipMemcpyHostToDevice));
    HIP(hipMemset(pa.data,0xff,(a_words+32)*4));
    auto pack=[&] {
        if(a_words) {
            pack_activation<<<(a_words+255)/256,256>>>(da.data+16,pa.data+16,o.m,o.k,mp/128,a_words);
            HIP(hipGetLastError());
        }
    };
    // Independently gate every packed word against the CPU logical packer.
    pack(); HIP(hipDeviceSynchronize());
    std::vector<uint32_t> packed(a_words+32);
    HIP(hipMemcpy(packed.data(),pa.data,packed.size()*4,hipMemcpyDeviceToHost));
    const auto expected=matrices.pack_operand(false,128,32);
    if(!std::equal(expected.begin(),expected.end(),packed.begin()+16)) throw std::runtime_error("GPU A pack mismatch");
    for(size_t i=0;i<16;++i) if(packed[i]!=0xffffffffu||packed[a_words+16+i]!=0xffffffffu)
        throw std::runtime_error("packed A guard modified");
    const auto r=reference(matrices);
    auto launch=[&] {
        native_gemm<<<blocks,256,16384>>>(row_major_a?reinterpret_cast<uint32_t*>(da.data+16):pa.data+16,
            db.data,dc.data+16,dt.data,o.m,o.n,mp,np,stages,0,ds.data);
        HIP(hipGetLastError());
    };
    std::fprintf(stderr,"HIP0 BDF=%s CUs%d residentCTA/CU%d layout=%s M/N/K=%d/%d/%d grid%d "
        "no model/context,automatic clocks,external NUMA; FP16 A/E4M3FN B/K128N128 FP32 scales\n",
        bdf,device.multiProcessorCount,occupancy,row_major_a?"rowmajor":"packed",o.m,o.n,o.k,blocks);
    std::fprintf(stderr,"wall_clock_khz=%d\n",rate);
    std::fprintf(stderr,"GPU A pack FULL %zu words exact; reference %s %zu outputs; A bytes rowmajor%zu packed%zu\n",
        a_words,r.full?"FULL":"SAMPLED",r.indices.size(),size_t(o.m)*o.k*2,a_words*4);
    launch(); HIP(hipDeviceSynchronize());
    const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(o.warmup_ms);
    while(std::chrono::steady_clock::now()<until) { launch();HIP(hipDeviceSynchronize()); }
    Event begin,packed_event,end;
    std::vector<float> c(size_t(o.m)*o.n+32);
    std::vector<uint64_t> stamps(size_t(blocks)*16);
    std::puts("layout,protocol,round,m,n,k,ctas,resident_ctas_per_cu,a_bytes,packed_a_bytes,pack_us,event_us,gemm_us,grid_body_us,median_core_MHz,useful_TFLOPs,executed_TFLOPs,reference_outputs,full_reference,valid");
    // Separate kernel-only and practical activation-path measurements. The
    // activation protocol includes GPU pack+launch for packed A, not CPU pack,
    // H2D, weight preprocessing, reference work or FP32->FP16 conversion.
    for(int round=0;round<o.rounds;++round) for(int p=0;p<2;++p) {
        const bool activation=(p^(round%2))!=0;
        HIP(hipMemset(dc.data,0xff,c.size()*4));
        HIP(hipEventRecord(begin.event));
        if(activation&&!row_major_a) { pack();HIP(hipEventRecord(packed_event.event)); }
        launch();HIP(hipEventRecord(end.event));HIP(hipEventSynchronize(end.event));
        float ms,pack_ms=0,gemm_ms;
        HIP(hipEventElapsedTime(&ms,begin.event,end.event));gemm_ms=ms;
        if(activation&&!row_major_a) {
            HIP(hipEventElapsedTime(&pack_ms,begin.event,packed_event.event));
            HIP(hipEventElapsedTime(&gemm_ms,packed_event.event,end.event));
        }
        HIP(hipMemcpy(c.data(),dc.data,c.size()*4,hipMemcpyDeviceToHost));verify(c,size_t(o.m)*o.n,r);
        HIP(hipMemcpy(stamps.data(),dt.data,stamps.size()*8,hipMemcpyDeviceToHost));
        const auto clock=clocks(stamps,blocks,rate,gemm_ms*1000);
        if(!o.prefix.empty()) {
            std::ofstream file(o.prefix+"-"+(activation?"activation":"kernel")+"-r"+std::to_string(round)+".bin",std::ios::binary);
            file.write(reinterpret_cast<const char*>(stamps.data()),stamps.size()*8);
            if(!file) throw std::runtime_error("clock dump failed");
        }
        std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f,%zu,%d,1\n",
            row_major_a?"rowmajor":"packed",activation?"activation":"kernel",round,o.m,o.n,o.k,blocks,occupancy,
            size_t(o.m)*o.k*2,a_words*4,pack_ms*1000,ms*1000,gemm_ms*1000,clock.body_us,clock.mhz,
            2.0*o.m*o.n*o.k/(ms*1e9),2.0*mp*np*o.k/(gemm_ms*1e9),r.indices.size(),int(r.full));
    }
    uint16_t guards[32];
    HIP(hipMemcpy(guards,da.data,32,hipMemcpyDeviceToHost));
    HIP(hipMemcpy(guards+16,da.data+16+size_t(o.m)*o.k,32,hipMemcpyDeviceToHost));
    for(auto value:guards) if(value!=0x7e00) throw std::runtime_error("row-major A guard modified");
    HIP(hipMemcpy(packed.data(),pa.data,packed.size()*4,hipMemcpyDeviceToHost));
    if(!std::equal(expected.begin(),expected.end(),packed.begin()+16)) throw std::runtime_error("packed A mutated");
    for(size_t i=0;i<16;++i) if(packed[i]!=0xffffffffu||packed[a_words+16+i]!=0xffffffffu)
        throw std::runtime_error("packed A guard mutated");
    std::fprintf(stderr,"COMPLETE all C finite after poison, logical refs/guards/input pack exact\n");
    return 0;
} catch(const std::exception& error) { std::fprintf(stderr,"ERROR %s\n",error.what());return 1; }
