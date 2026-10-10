// CU/SIMD placement and overlapping wave intervals for the A/B/C readers.
// gfx90a only: HW_ID fields are documented in HIP amd_device_functions.h.
#include "readers.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)

struct Stamp { uint64_t start,end; uint32_t hw_start,hw_end; };

__device__ __forceinline__ uint32_t hardware_id() {
  // HW_REG_HW_ID=4, offset=0, width=32; encoding uses width minus one.
  return __builtin_amdgcn_s_getreg((31u<<11)|4u);
}

template<class Reader>
__global__ __launch_bounds__(256,1)
void read_placed(const float* __restrict__ input,size_t count,uint32_t* __restrict__ sink,
                 Stamp* __restrict__ stamps) {
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const uint64_t start=wall_clock64();
  const uint32_t hw_start=hardware_id();
  if((threadIdx.x&63)==0) { stamps[wave].start=start; stamps[wave].hw_start=hw_start; }
  read_body<Reader,4>(input,count,sink);
  const uint64_t end=wall_clock64();
  const uint32_t hw_end=hardware_id();
  if((threadIdx.x&63)==0) { stamps[wave].end=end; stamps[wave].hw_end=hw_end; }
}

struct Case {
  std::string name;
  const void* marked;
  int original_regs,marked_regs,original_capacity,marked_capacity;
};
template<class Reader> void add(std::vector<Case>& cases) {
  const void* original=reinterpret_cast<const void*>(read_tiles<Reader,4>);
  const void* marked=reinterpret_cast<const void*>(read_placed<Reader>);
  hipFuncAttributes a{},b{}; HIP(hipFuncGetAttributes(&a,original)); HIP(hipFuncGetAttributes(&b,marked));
  if(a.localSizeBytes || b.localSizeBytes || a.sharedSizeBytes || b.sharedSizeBytes)
    throw std::runtime_error("unexpected scratch or LDS");
  int original_capacity,marked_capacity;
  HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&original_capacity,original,256,0));
  HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&marked_capacity,marked,256,0));
  cases.push_back({Reader::name,marked,a.numRegs,b.numRegs,original_capacity,marked_capacity});
}

struct Context {
  hipStream_t stream{};
  float* input{}; uint32_t* sink{}; Stamp* stamps{};
  size_t count=(size_t(1)<<30)/4;
  Context(const std::string& out) {
    HIP(hipSetDevice(0));
    hipDeviceProp_t p{}; HIP(hipGetDeviceProperties(&p,0));
    if(p.multiProcessorCount!=104 || p.warpSize!=64 || std::string(p.gcnArchName).find("gfx90a")!=0)
      throw std::runtime_error("packing decoder is gfx90a/104-CU/wave64 only");
    HIP(hipStreamCreate(&stream)); HIP(hipMalloc(&input,count*4));
    HIP(hipMalloc(&sink,1664*4*4)); HIP(hipMalloc(&stamps,1664*4*sizeof(Stamp)));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    int rate; HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));
    if(rate<=0) throw std::runtime_error("steady clock unavailable");
    std::ofstream metadata(out+"/device.txt");
    metadata<<"device=0 PCI="<<pci<<" name="<<p.name<<" arch="<<p.gcnArchName
            <<" CUs="<<p.multiProcessorCount<<" wave="<<p.warpSize<<" wall_clock_khz="<<rate
            <<" max_threads_per_CU="<<p.maxThreadsPerMultiProcessor<<'\n';
    initialize<<<832,256,0,stream>>>(input,count);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  }
  ~Context() {
    (void)hipFree(stamps); (void)hipFree(sink); (void)hipFree(input); (void)hipStreamDestroy(stream);
  }
  void launch(const Case& c,int grid) {
    void* args[]={&input,&count,&sink,&stamps};
    HIP(hipLaunchKernel(c.marked,dim3(grid),dim3(256),args,0,stream));
  }
  void collect(const Case& c,int grid,const std::string& path) {
    HIP(hipMemsetAsync(stamps,0,grid*4*sizeof(Stamp),stream));
    launch(c,grid);
    std::vector<uint32_t> sums(grid*4);
    std::vector<Stamp> records(grid*4);
    HIP(hipMemcpyAsync(sums.data(),sink,sums.size()*4,hipMemcpyDeviceToHost,stream));
    HIP(hipMemcpyAsync(records.data(),stamps,records.size()*sizeof(Stamp),hipMemcpyDeviceToHost,stream));
    HIP(hipStreamSynchronize(stream));
    uint32_t sum=0; for(auto word:sums) sum+=word;
    if(sum!=expected_sum(count)) throw std::runtime_error("checksum mismatch");
    std::ofstream raw(path);
    raw<<"CTA,wave,start_tick,end_tick,hw_start,hw_end,CU_key,SE,SH,CU_local,SIMD,slot\n";
    for(int cta=0;cta<grid;++cta) {
      const uint32_t cu=(records[cta*4].hw_start>>8)&255;
      for(int w=0;w<4;++w) {
        const auto& r=records[cta*4+w];
        // Compare CU, SIMD and slot. Do not confuse TG/queue/state fields
        // with physical placement, and do not silently ignore migration.
        constexpr uint32_t placement_mask=0xff3f;
        if(r.end<=r.start || ((r.hw_start^r.hw_end)&placement_mask) || ((r.hw_start>>8)&255)!=cu)
          throw std::runtime_error("invalid timestamps or placement changed");
        const uint32_t hw=r.hw_start;
        raw<<cta<<','<<w<<','<<r.start<<','<<r.end<<','<<hw<<','<<r.hw_end<<','<<cu
           <<','<<((hw>>13)&7)<<','<<((hw>>12)&1)<<','<<((hw>>8)&15)
           <<','<<((hw>>4)&3)<<','<<(hw&15)<<'\n';
      }
    }
  }
};

int main(int argc,char** argv) try {
  if(argc<2 || argc>3) throw std::runtime_error("usage: packing OUT [SAMPLES=7]");
  const std::string out=argv[1]; const int samples=argc>2?std::stoi(argv[2]):7;
  if(samples<3) throw std::runtime_error("samples >=3");
  std::filesystem::create_directories(out);
  Context context(out);
  std::vector<Case> cases; add<A>(cases); add<B>(cases); add<C>(cases);
  std::ofstream resources(out+"/resources.csv");
  resources<<"pattern,original_regs,marked_regs,original_CTAs_per_CU,marked_CTAs_per_CU,original_device_capacity,marked_device_capacity\n";
  for(const auto& c:cases) {
    resources<<c.name<<','<<c.original_regs<<','<<c.marked_regs<<','<<c.original_capacity<<','<<c.marked_capacity
             <<','<<104*c.original_capacity<<','<<104*c.marked_capacity<<'\n';
    std::fprintf(stderr,"%s: capacity original/marked=%d/%d CTAs per CU\n",c.name.c_str(),c.original_capacity,c.marked_capacity);
  }
  // No artificial holds, atomics, CTA/grid barriers or cooperative launches.
  // The real read body lasts long enough for admission overlap to be observed.
  for(int i=0;i<100;++i) context.launch(cases.front(),104);
  HIP(hipStreamSynchronize(context.stream));
  struct Work { size_t reader; int grid; };
  std::vector<Work> work;
  for(size_t i=0;i<cases.size();++i) for(int grid:{52,104,208,416,832,1664}) work.push_back({i,grid});
  std::mt19937 random(0x90a);
  std::ofstream manifest(out+"/manifest.csv");
  manifest<<"pattern,grid,block,chunks,bytes,sample,trace\n";
  for(int sample=0;sample<samples;++sample) {
    std::shuffle(work.begin(),work.end(),random);
    for(auto w:work) {
      const auto& c=cases[w.reader];
      context.launch(c,w.grid); context.launch(c,w.grid);
      const auto filename=c.name+"-g"+std::to_string(w.grid)+"-s"+std::to_string(sample)+".csv";
      context.collect(c,w.grid,out+"/"+filename);
      manifest<<c.name<<','<<w.grid<<",256,4,"<<context.count*4<<','<<sample<<','<<filename<<'\n';
    }
    manifest.flush(); std::fprintf(stderr,"sample %d/%d complete\n",sample+1,samples);
  }
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
