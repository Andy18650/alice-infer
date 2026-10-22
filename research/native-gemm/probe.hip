// Standalone native gfx90a FP16 GEMM. No BLAS/CK or other GEMM library.
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
#include "data.hpp"
#define HIP(x) do{auto status_=(x);if(status_!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status_));}while(0)
using F16=float __attribute__((ext_vector_type(16)));
#include "compute.hpp"
__global__ __launch_bounds__(256) void native_gemm(const uint32_t* a,const uint32_t* b,float* out,uint64_t* stamps,int m,int n,int mp,int np,int tiles,int swizzle){
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  const int gn=np/kTile,mi=blockIdx.x/gn,ni=blockIdx.x%gn;
  const int wave=threadIdx.x/64,lane=threadIdx.x%64,wm=wave/2,wn=wave%2;
  const int half=kTile/2,panel_bytes=kTile*kStageK,record_bytes=kTile*8;
  const uintptr_t source=reinterpret_cast<uintptr_t>(wave<2?a:b)+size_t((wave<2?mi:ni)*2+wave%2)*panel_bytes+lane*16;
  const uint32_t stride=__builtin_amdgcn_readfirstlane(uint32_t((wave<2?mp:np)*kStageK*2-(kQ*(kTile/128)/4-1)*4096));
  const uint32_t base=uint32_t(reinterpret_cast<uintptr_t>(smem));
  const int lds_lane=swizzle?(lane^(lane>>3)):lane;
  const uint32_t dst=base+(wave<2?wave*record_bytes:kTile*kStageK*2+(wave-2)*record_bytes)+lds_lane*16;
  const uint32_t alo=base+wm*record_bytes+lds_lane*16,blo=base+kTile*kStageK*2+wn*record_bytes+lds_lane*16;
  F16 c[kR*kR]={};uint64_t start,end,ws,we;
  pipeline(uint32_t(source),uint32_t(source>>32),stride,dst,alo,blo,tiles,c,start,end,ws,we);
  if(lane==0){auto s=stamps+(size_t(blockIdx.x)*4+wave)*4;s[0]=start;s[1]=end;s[2]=ws;s[3]=we;}
#pragma unroll
  for(int i=0;i<kR;++i){
#pragma unroll
    for(int j=0;j<kR;++j){
#pragma unroll
      for(int r=0;r<16;++r){
        int row=mi*kTile+wm*half+i*32+8*(r/4)+4*(lane/32)+r%4;
        int col=ni*kTile+wn*half+j*32+lane%32;
        if(row<m&&col<n)out[size_t(row)*n+col]=c[i*kR+j][r];
      }
    }
  }
}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){HIP(hipMalloc(&p,std::max(size_t(1),n)*sizeof(T)));}~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;};
struct Events{hipEvent_t b,e;Events(){HIP(hipEventCreate(&b));HIP(hipEventCreate(&e));}~Events(){(void)hipEventDestroy(b);(void)hipEventDestroy(e);}};
int main(int argc,char** argv)try{
  int m=1664,n=1024,k=8192,rounds=9,warm_ms=200,swizzle=0;uint32_t sa=1729,sb=2718;std::string stamp_prefix;
  for(int x=1;x<argc;++x){std::string key=argv[x];if(key=="--help"){std::printf("--m N --n N --k N --rounds N --warmup-ms N --seed-a N --seed-b N --swizzle 0|1 --stamps-prefix PATH; compiled tile%d,StageK%d,FP16 packed panels,FP32 C; native only\n",kTile,kStageK);return 0;}if(++x==argc)throw std::runtime_error("missing value");std::string v=argv[x];
    if(key=="--m")m=std::stoi(v);else if(key=="--n")n=std::stoi(v);else if(key=="--k")k=std::stoi(v);else if(key=="--rounds")rounds=std::stoi(v);else if(key=="--warmup-ms")warm_ms=std::stoi(v);else if(key=="--stamps-prefix")stamp_prefix=v;else if(key=="--swizzle")swizzle=std::stoi(v);else if(key=="--seed-a")sa=uint32_t(std::stoul(v));else if(key=="--seed-b")sb=uint32_t(std::stoul(v));else throw std::runtime_error("unknown option "+key);}
  if(m<1||n<1||m>16384||n>16384||k<0||k>65536||rounds<1||warm_ms<0||(swizzle!=0&&swizzle!=1))throw std::runtime_error("dimensions/options out of range");
  HIP(hipSetDevice(0));int wallrate;HIP(hipDeviceGetAttribute(&wallrate,hipDeviceAttributeWallClockRate,0));hipDeviceProp_t props;HIP(hipGetDeviceProperties(&props,0));char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
  int mp=(m+kTile-1)/kTile*kTile,np=(n+kTile-1)/kTile*kTile,kt=(k+kStageK-1)/kStageK,blocks=mp/kTile*(np/kTile),resident=0;
  HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,native_gemm,256,kLdsBytes));
  std::fprintf(stderr,"kind=%s\n",kResident?"resident-control":"gemm");
  if(kResident)std::fprintf(stderr,"Resident control repeats first padded stage fragment, not ordinary full GEMM\n");
  std::fprintf(stderr,"LDS lane permutation=%s (address-only, producer/consumer agree)\n",swizzle?"lane XOR lane>>3":"linear");
  Data data(m,n,k,sa,sb);auto pack_start=std::chrono::steady_clock::now();auto ap=data.pack(false,kTile,kStageK),bp=data.pack(true,kTile,kStageK);double pack_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-pack_start).count();
  Buffer<uint32_t> da(ap.size()),db(bp.size());Buffer<float> output(size_t(m)*n+32);Buffer<uint64_t> stamps(size_t(blocks)*16);
  if(!ap.empty())HIP(hipMemcpy(da.p,ap.data(),ap.size()*4,hipMemcpyHostToDevice));if(!bp.empty())HIP(hipMemcpy(db.p,bp.data(),bp.size()*4,hipMemcpyHostToDevice));
  std::fprintf(stderr,"HIP0 BDF=%s O3 gfx90a, CUs=%d tile=%d stageK=%d singleLDS=%d residentCTA/CU=%d grid=%dx%d(%d) block256 M/N/K=%d/%d/%d padded=%d/%d/%d; FP16 packed/FP32 C,packing %.3f ms outside timing,no BLAS/CK,model/context=none,external NUMA,no clock override\n",bdf,props.multiProcessorCount,kTile,kStageK,kLdsBytes,resident,mp/kTile,np/kTile,blocks,m,n,k,mp,np,kt*kStageK,pack_ms);
  // Independent reference over original row-major integer operands, not packed
  // fragments. Exact FP32 sums for K<=65536, all inputs are integer/16.
  std::vector<size_t> indices;
  bool full=uint64_t(m)*n*std::max(k,1)<=30000000;
  if(full){indices.resize(size_t(m)*n);for(size_t q=0;q<indices.size();++q)indices[q]=q;}
  else{
    std::mt19937 rng(314159);for(int q=0;q<2048;++q)indices.push_back(size_t(rng())%(size_t(m)*n));
    for(int i=0;i<m;i+=32)for(int j:{0,n-1})indices.push_back(size_t(i)*n+j);
    for(int j=0;j<n;j+=32)for(int i:{0,m-1})indices.push_back(size_t(i)*n+j);
    for(int i:{0,m-1})for(int j:{0,n-1})indices.push_back(size_t(i)*n+j);
    std::sort(indices.begin(),indices.end());indices.erase(std::unique(indices.begin(),indices.end()),indices.end());
  }
  std::vector<float> ref;ref.reserve(indices.size());for(size_t i:indices)ref.push_back(kResident?data.ref(int(i/n),int(i%n),std::min(k,kStageK))*kt:data.ref(int(i/n),int(i%n)));
  std::fprintf(stderr,"REFERENCE %s %zu outputs from original row-major operands; all C scanned finite/guards every round\n",full?"FULL":"SAMPLED",indices.size());
  auto launch=[&]{native_gemm<<<blocks,256,kLdsBytes>>>(da.p,db.p,output.p+16,stamps.p,m,n,mp,np,kt,swizzle);HIP(hipGetLastError());};
  launch();HIP(hipDeviceSynchronize());auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warm_ms);
  while(std::chrono::steady_clock::now()<until){launch();HIP(hipDeviceSynchronize());}
  Events events;std::vector<float> y(size_t(m)*n+32);std::vector<uint64_t> ts(size_t(blocks)*16);
  std::puts("kind,round,tile,stage_k,swizzle,m,n,k,mp,np,kp,ctas,resident_ctas_per_cu,lds_bytes,event_us,grid_body_us,useful_TFLOPs,executed_TFLOPs,median_core_MHz,median_cta_share_of_cu_peak,global_clock_adjusted_efficiency,reference_outputs,full_reference,mismatches,nonfinite,guards,valid,pack_ms");
  for(int r=0;r<rounds;++r){HIP(hipMemset(output.p,0xff,y.size()*4));HIP(hipEventRecord(events.b));launch();HIP(hipEventRecord(events.e));HIP(hipEventSynchronize(events.e));float ms;HIP(hipEventElapsedTime(&ms,events.b,events.e));HIP(hipMemcpy(y.data(),output.p,y.size()*4,hipMemcpyDeviceToHost));HIP(hipMemcpy(ts.data(),stamps.p,ts.size()*8,hipMemcpyDeviceToHost));
    int bad=0,nonfinite=0,guards=0;for(size_t q=0;q<size_t(m)*n;++q)nonfinite+=!std::isfinite(y[q+16]);
    for(size_t q=0;q<indices.size();++q)if(y[indices[q]+16]!=ref[q]){if(bad<4)std::fprintf(stderr,"MISMATCH %zu expected %.9g actual %.9g\n",indices[q],ref[q],y[indices[q]+16]);++bad;}
    for(int q=0;q<16;++q)guards+=(std::bit_cast<uint32_t>(y[q])!=0xffffffffu)+(std::bit_cast<uint32_t>(y[size_t(m)*n+16+q])!=0xffffffffu);
    uint64_t first=UINT64_MAX,last=0;std::vector<double> clocks,util;
    const double per_cta_flops=2.0*kTile*kTile*kt*kStageK;
    for(int c=0;c<blocks;++c){auto s=ts.data()+size_t(c)*16;uint64_t b=UINT64_MAX,e=0,wb=UINT64_MAX,we=0;for(int w=0;w<4;++w){b=std::min(b,s[w*4]);e=std::max(e,s[w*4+1]);wb=std::min(wb,s[w*4+2]);we=std::max(we,s[w*4+3]);}first=std::min(first,wb);last=std::max(last,we);double us=double(we-wb)*1000/wallrate;clocks.push_back(double(e-b)/us);util.push_back(per_cta_flops/(double(e-b)*1024));}
    if(!stamp_prefix.empty()){std::ofstream file(stamp_prefix+"-r"+std::to_string(r)+".bin",std::ios::binary);file.write(reinterpret_cast<const char*>(ts.data()),std::streamsize(ts.size()*8));if(!file)throw std::runtime_error("timestamp output failed");}
    std::sort(clocks.begin(),clocks.end());std::sort(util.begin(),util.end());auto med=[](const std::vector<double>& v){return(v[(v.size()-1)/2]+v[v.size()/2])/2;};
    std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.9f,%.9f,%zu,%d,%d,%d,%d,%d,%.3f\n",kResident?"resident-control":"gemm",r,kTile,kStageK,swizzle,m,n,k,mp,np,kt*kStageK,blocks,resident,kLdsBytes,ms*1000,double(last-first)*1000/wallrate,2.0*m*n*k/(ms*1e9),2.0*mp*np*kt*kStageK/(ms*1e9),med(clocks),med(util),(2.0*m*n*k/(ms*1e9))/(props.multiProcessorCount*1024.0*med(clocks)/1e6),indices.size(),full,bad,nonfinite,guards,!bad&&!nonfinite&&!guards,pack_ms);
    if(bad||nonfinite||guards)throw std::runtime_error("GEMM verification failed");}
  std::fprintf(stderr,"COMPLETE all reference outputs/finite C/guards exact\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
