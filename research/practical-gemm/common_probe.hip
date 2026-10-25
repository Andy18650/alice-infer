// One FP8 B allocation, two consumers: row-major-A GEMM and K-split GEMV.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include "common_host.hpp"
#include "compute.hpp"
#include "kernel.hpp"
#include "common_gpu.hpp"

constexpr const char* formats[]={"native","plain","block128","block256"};
struct Options {
    int m=128,n=2560,k=8192,rounds=9,warmup=200,scale=2,finite=0;
    uint32_t sa=1729,sb=2718;
    std::string path="gemm",prefix;
};
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];if(++i==argc) throw std::runtime_error("missing value");
        const std::string value=argv[i];
        if(key=="--m")o.m=std::stoi(value);
        else if(key=="--n")o.n=std::stoi(value);
        else if(key=="--k")o.k=std::stoi(value);
        else if(key=="--rounds")o.rounds=std::stoi(value);
        else if(key=="--warmup-ms")o.warmup=std::stoi(value);
        else if(key=="--scale-pattern")o.scale=std::stoi(value);
        else if(key=="--b-pattern")o.finite=std::stoi(value);
        else if(key=="--seed-a")o.sa=uint32_t(std::stoul(value));
        else if(key=="--seed-b")o.sb=uint32_t(std::stoul(value));
        else if(key=="--path")o.path=value;
        else if(key=="--stamps-prefix")o.prefix=value;
        else throw std::runtime_error("unknown option "+key);
    }
    if(o.m<1||o.m>16384||o.n<1||o.n>32768||o.k<0||o.k>65536||o.k%32
       ||o.rounds<1||o.warmup<0||o.scale<0||o.scale>3||o.finite<0||o.finite>1
       ||(o.path!="gemm"&&o.path!="decode")||(o.path=="decode"&&o.m!=1))
        throw std::runtime_error("requires K multiple-of32 or zero; decode M=1; invalid options");
    return o;
}

void primitives() {
    constexpr int count=254*4*16;
    auto finite=[](unsigned i){i%=254;return uint32_t(i<127?i:i+1);};
    std::vector<uint32_t> words(count);
    for(int code=0;code<254;++code) for(int position=0;position<4;++position) for(int neighbor=0;neighbor<16;++neighbor) {
        uint32_t value=0;
        for(int byte=0;byte<4;++byte) value|=finite(byte==position?code:unsigned(code*73+neighbor*17+byte*31))<<(byte*8);
        words[(code*4+position)*16+neighbor]=value;
    }
    Buffer<uint32_t> di(count),dh(count*2);Buffer<uint4> dt(count);
    HIP(hipMemcpy(di.data,words.data(),count*4,hipMemcpyHostToDevice));
    primitive_check<<<(count+255)/256,256>>>(di.data,dh.data,dt.data,count);
    HIP(hipGetLastError());HIP(hipDeviceSynchronize());
    std::vector<uint32_t> halves(count*2);std::vector<uint4> transposed(count);
    HIP(hipMemcpy(halves.data(),dh.data,count*8,hipMemcpyDeviceToHost));
    HIP(hipMemcpy(transposed.data(),dt.data,count*16,hipMemcpyDeviceToHost));
    for(int i=0;i<count;++i) {
        for(int byte=0;byte<4;++byte)
            if(uint16_t(halves[2*i+byte/2]>>(16*(byte%2)))!=fp8_reference_half(uint8_t(words[i]>>(8*byte))))
                throw std::runtime_error("finite FP8 expansion primitive mismatch");
        const uint32_t actual[4]={transposed[i].x,transposed[i].y,transposed[i].z,transposed[i].w};
        for(int col=0;col<4;++col) {
            uint32_t expected=0;
            for(int row=0;row<4;++row) expected|=((words[(i+row)%count]>>(col*8))&255)<<(row*8);
            if(expected!=actual[col]) throw std::runtime_error("byte transpose primitive mismatch");
        }
    }
    std::fprintf(stderr,"PRIMITIVES 65024 exact half outputs;16256 byte transposes PASS\n");
}

int main(int argc,char** argv) try {
    const auto o=parse(argc,argv);
    HIP(hipSetDevice(0));
    hipDeviceProp_t device;HIP(hipGetDeviceProperties(&device,0));
    char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
    int rate;HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));
    if(std::string(bdf)!="0000:31:00.0"||device.multiProcessorCount!=104)throw std::runtime_error("wrong GPU");
    primitives();
    const int mp=(o.m+127)/128*128,np=(o.n+127)/128*128,stages=o.k/32,groups=(o.k+127)/128;
    constexpr int bn=COMMON_B_FORMAT==3?256:128;
    const int storage_n=(o.n+bn-1)/bn*bn;
    const int storage_k=COMMON_B_FORMAT>=2?(o.k+127)/128*128:o.k;
    const int gemm_ctas=(mp/128)*(np/128),decode_ctas=((np/4)*std::max(1,groups)+63)/64;
    const bool decode=o.path=="decode";
    const int ctas=decode?decode_ctas:gemm_ctas,waves=decode?1:4;
    int gemm_occupancy,decode_occupancy;
    HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&gemm_occupancy,native_gemm,256,16384));
    HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&decode_occupancy,decode_partial,64,0));
    ScaledMatrices matrices(o.m,o.n,o.k,o.sa,o.sb);matrices.set_finite_pattern(o.finite!=0);matrices.set_scales(o.scale,true);
    std::vector<uint16_t> a(size_t(o.m)*o.k+32,0x7e00);
    for(size_t i=0;i<matrices.row_major_a_units.size();++i)a[16+i]=half_bits_from_unit(matrices.row_major_a_units[i]);
    const size_t bytes=size_t(storage_n)*storage_k;
    std::vector<uint8_t> b(bytes+512,0x7f),logical(bytes);
    for(int k=0;k<storage_k;++k) for(int n=0;n<storage_n;++n) {
        const uint8_t value=k<o.k&&n<o.n?matrices.b_code(k,n):0;
        b[256+cpu_b_index(k,n,storage_n)]=value;logical[size_t(k)*storage_n+n]=value;
    }
    Buffer<uint16_t> da(a.size());Buffer<uint8_t> db(b.size()),du(bytes);
    Buffer<float> ds(matrices.scales.size()+32),dc(size_t(o.m)*o.n+32),dd(o.n+32);
    const size_t partial_count=size_t(np)*std::max(1,groups);
    Buffer<float> dp(partial_count+32);
    Buffer<uint64_t> dt(std::max(size_t(gemm_ctas)*16,size_t(decode_ctas)*4));
    HIP(hipMemcpy(da.data,a.data(),a.size()*2,hipMemcpyHostToDevice));
    HIP(hipMemcpy(db.data,b.data(),b.size(),hipMemcpyHostToDevice));
    HIP(hipMemset(ds.data,0xff,(matrices.scales.size()+32)*4));
    if(!matrices.scales.empty())HIP(hipMemcpy(ds.data+16,matrices.scales.data(),matrices.scales.size()*4,hipMemcpyHostToDevice));
    if(bytes) {
        unpack_b<<<(bytes+255)/256,256>>>(db.data+256,du.data,storage_n,storage_k);
        HIP(hipGetLastError());HIP(hipDeviceSynchronize());
        std::vector<uint8_t> unpacked(bytes);HIP(hipMemcpy(unpacked.data(),du.data,bytes,hipMemcpyDeviceToHost));
        if(unpacked!=logical)throw std::runtime_error("full B layout roundtrip mismatch");
    }
    const auto rg=make_reference(matrices),rd=o.m==1?rg:make_reference(matrices,true);
    auto gemm=[&] {
        native_gemm<<<gemm_ctas,256,16384>>>(reinterpret_cast<uint32_t*>(da.data+16),
            reinterpret_cast<uint32_t*>(db.data+256),dc.data+16,dt.data,o.m,o.n,mp,np,stages,0,ds.data+16);
        HIP(hipGetLastError());
    };
    auto gemv=[&] {
        decode_partial<<<decode_ctas,64>>>(db.data+256,da.data+16,dp.data+16,dt.data,np,o.k,storage_n);
        HIP(hipGetLastError());
        decode_combine<<<(o.n+255)/256,256>>>(dp.data+16,ds.data+16,dd.data+16,o.n,np,groups);
        HIP(hipGetLastError());
    };
    std::vector<float> cg(size_t(o.m)*o.n+32),cd(o.n+32);
    HIP(hipMemset(dc.data,0xff,cg.size()*4));HIP(hipMemset(dd.data,0xff,cd.size()*4));
    HIP(hipMemset(dp.data,0xff,(partial_count+32)*4));
    gemm();gemv();HIP(hipDeviceSynchronize());
    HIP(hipMemcpy(cg.data(),dc.data,cg.size()*4,hipMemcpyDeviceToHost));verify(cg,size_t(o.m)*o.n,rg);
    HIP(hipMemcpy(cd.data(),dd.data,cd.size()*4,hipMemcpyDeviceToHost));verify(cd,o.n,rd);
    std::fprintf(stderr,"INTEROP same B pointer GEMM+decode PASS;full layout roundtrip%zu bytes\n",bytes);
    std::fprintf(stderr,"HIP0 BDF=%s CUs%d format=%s path=%s M/N/K=%d/%d/%d grid%d waves%d occupancy%d "
        "wall_clock_khz=%d; automatic clocks,no model/context,external NUMA\n",bdf,device.multiProcessorCount,
        formats[COMMON_B_FORMAT],o.path.c_str(),o.m,o.n,o.k,ctas,waves,decode?decode_occupancy:gemm_occupancy,rate);
    auto launch=[&] {if(decode)gemv();else gemm();};
    launch();HIP(hipDeviceSynchronize());
    const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(o.warmup);
    while(std::chrono::steady_clock::now()<until){launch();HIP(hipDeviceSynchronize());}
    Event begin,end;
    std::vector<uint64_t> stamps(size_t(ctas)*waves*4);
    const auto& ref=decode?rd:rg;
    auto& output=decode?cd:cg;
    const size_t count=decode?size_t(o.n):size_t(o.m)*o.n;
    float* destination=decode?dd.data:dc.data;
    std::puts("format,path,round,m,n,k,ctas,waves_per_cta,resident_ctas_per_cu,b_storage_bytes,event_us,grid_body_us,median_core_MHz,useful_TFLOPs,useful_weight_GBs,requested_weight_GBs,reference_outputs,full_reference,valid");
    for(int round=0;round<o.rounds;++round) {
        HIP(hipMemset(destination,0xff,output.size()*4));
        HIP(hipMemset(dt.data,0xff,stamps.size()*8));
        if(decode)HIP(hipMemset(dp.data,0xff,(partial_count+32)*4));
        HIP(hipEventRecord(begin.event));launch();HIP(hipEventRecord(end.event));HIP(hipEventSynchronize(end.event));
        float ms;HIP(hipEventElapsedTime(&ms,begin.event,end.event));
        HIP(hipMemcpy(output.data(),destination,output.size()*4,hipMemcpyDeviceToHost));verify(output,count,ref);
        HIP(hipMemcpy(stamps.data(),dt.data,stamps.size()*8,hipMemcpyDeviceToHost));
        const auto clock=check_clocks(stamps,ctas,waves,rate,ms*1000);
        if(!o.prefix.empty()) {
            std::ofstream file(o.prefix+"-r"+std::to_string(round)+".bin",std::ios::binary);
            file.write(reinterpret_cast<const char*>(stamps.data()),stamps.size()*8);
            if(!file)throw std::runtime_error("clock dump failed");
        }
        std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f,%zu,%d,1\n",
            formats[COMMON_B_FORMAT],o.path.c_str(),round,o.m,o.n,o.k,ctas,waves,decode?decode_occupancy:gemm_occupancy,
            bytes,ms*1000,clock.body_us,clock.mhz,2.0*o.m*o.n*o.k/(ms*1e9),
            double(o.n)*o.k/(ms*1e6),double(np)*o.k/(ms*1e6),ref.indices.size(),int(ref.full));
    }
    // Check immutable B bytes (not merely guards) after BOTH consumers ran.
    std::vector<uint8_t> after(b.size());HIP(hipMemcpy(after.data(),db.data,b.size(),hipMemcpyDeviceToHost));
    if(after!=b)throw std::runtime_error("B payload/guard mutated");
    uint16_t ag[32];HIP(hipMemcpy(ag,da.data,32,hipMemcpyDeviceToHost));
    HIP(hipMemcpy(ag+16,da.data+16+size_t(o.m)*o.k,32,hipMemcpyDeviceToHost));
    for(auto value:ag)if(value!=0x7e00)throw std::runtime_error("A guard mutated");
    uint32_t guards[32];
    for(auto pair:{std::pair<float*,size_t>(dp.data,partial_count),{ds.data,matrices.scales.size()}}) {
        HIP(hipMemcpy(guards,pair.first,64,hipMemcpyDeviceToHost));
        HIP(hipMemcpy(guards+16,pair.first+16+pair.second,64,hipMemcpyDeviceToHost));
        for(auto value:guards)if(value!=0xffffffffu)throw std::runtime_error("scratch/scale guard mutated");
    }
    std::fprintf(stderr,"COMPLETE timed refs/finite C/guards/B immutable;interop GEMM+decode pass\n");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
