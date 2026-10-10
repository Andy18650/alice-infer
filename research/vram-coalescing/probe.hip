// A/B/C: same 256-dword wave tiles, different per-instruction addresses.
// Native gfx90a HIP/C++20, uint32_t input and built-in uint4 loads/checksums.
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

struct Case {
  std::string pattern;
  int chunks,grid;
  const void* function;
  int registers,capacity;
  size_t scratch;
};
template<class Reader,int Chunks>
void add(std::vector<Case>& cases,int grid) {
  const void* function=reinterpret_cast<const void*>(read_tiles<Reader,Chunks>);
  hipFuncAttributes attributes{}; HIP(hipFuncGetAttributes(&attributes,function));
  int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,function,256,0));
  cases.push_back({Reader::name,Chunks,grid,function,attributes.numRegs,capacity,attributes.localSizeBytes});
}
template<int Chunks> void add_abc(std::vector<Case>& cases,int grid) {
  add<A,Chunks>(cases,grid); add<B,Chunks>(cases,grid); add<C,Chunks>(cases,grid);
}

struct Context {
  hipDeviceProp_t device{};
  hipStream_t stream{};
  hipEvent_t start{},end{};
  uint32_t* input{}; uint32_t* sink{};
  Context(const std::string& out) {
    HIP(hipSetDevice(0)); HIP(hipGetDeviceProperties(&device,0));
    if(device.warpSize!=64 || device.multiProcessorCount!=104) throw std::runtime_error("requires 104-CU wave64 device");
    HIP(hipStreamCreate(&stream)); HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
    HIP(hipMalloc(&input,size_t(4)<<30)); HIP(hipMalloc(&sink,832*4*4));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    std::ofstream meta(out+"/device.txt");
    meta<<"device=0 PCI="<<pci<<" name="<<device.name<<" arch="<<device.gcnArchName
        <<" CUs="<<device.multiProcessorCount<<" wave="<<device.warpSize
        <<" pointer="<<static_cast<void*>(input)<<" input=uint32_t vector=uint4 generator=affine_mod2^32 scheduling_barriers="<<RESEARCH_SCHED_BARRIERS<<'\n'; meta.close();
    initialize<<<832,256,0,stream>>>(input,(size_t(4)<<30)/4);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  }
  ~Context() {
    (void)hipFree(sink); (void)hipFree(input); (void)hipEventDestroy(start);
    (void)hipEventDestroy(end); (void)hipStreamDestroy(stream);
  }
  void launch(const Case& c,size_t bytes) {
    size_t count=bytes/4; void* args[]={&input,&count,&sink};
    HIP(hipLaunchKernel(c.function,dim3(c.grid),dim3(256),args,0,stream));
  }
  void validate(const Case& c,size_t bytes) {
    std::vector<uint32_t> host(c.grid*4);
    HIP(hipMemcpyAsync(host.data(),sink,host.size()*4,hipMemcpyDeviceToHost,stream)); HIP(hipStreamSynchronize(stream));
    uint32_t sum=0; for(auto value:host) sum+=value;
    if(sum!=expected_sum(bytes/4)) throw std::runtime_error("checksum mismatch: "+c.pattern);
  }
  double measure(const Case& c,size_t bytes) {
    HIP(hipEventRecord(start,stream));
    for(int i=0;i<12;++i) launch(c,bytes);
    HIP(hipGetLastError()); HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
    float ms; HIP(hipEventElapsedTime(&ms,start,end)); return double(ms)/12;
  }
};

int main(int argc,char** argv) try {
  if(argc<2 || argc>4) throw std::runtime_error("usage: probe OUT [quick|sweep|blocks] [ROUNDS=9]");
  const std::string out=argv[1],phase=argc>2?argv[2]:"quick";
  const int rounds=argc>3?std::stoi(argv[3]):9;
  if((phase!="quick" && phase!="sweep" && phase!="blocks") || rounds<3 || rounds%2==0) throw std::runtime_error("invalid phase/rounds");
  std::filesystem::create_directories(out);
  Context context(out);
  std::vector<Case> cases;
  if(phase=="blocks") {
    for(int grid:{104,208,416}) { add<A,4>(cases,grid); add<B,4>(cases,grid); }
  } else {
    add_abc<1>(cases,104);
    if(phase=="sweep") {
      add_abc<2>(cases,104); add_abc<4>(cases,104); add_abc<8>(cases,104);
      add_abc<1>(cases,832); add_abc<2>(cases,832); add_abc<4>(cases,832); add_abc<8>(cases,832);
    }
  }
  std::ofstream resources(out+"/resources.csv");
  resources<<"pattern,chunks,grid,block,reported_regs,capacity_CTAs_per_CU,private_bytes\n";
  for(const auto& c:cases) {
    resources<<c.pattern<<','<<c.chunks<<','<<c.grid<<",256,"<<c.registers<<','<<c.capacity<<','<<c.scratch<<'\n';
    if(c.scratch) throw std::runtime_error("scratch-using candidate: "+c.pattern);
    for(size_t bytes:{size_t(64),size_t(4160),size_t((1<<20)+64)}) {
      context.launch(c,bytes); HIP(hipGetLastError()); context.validate(c,bytes);
    }
  }
  std::fprintf(stderr,"%zu cases passed all short/tail checks\n",cases.size());
  for(int i=0;i<250;++i) context.launch(cases.front(),size_t(1)<<30);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(context.stream));
  struct Work { size_t index,bytes; };
  std::vector<Work> work;
  for(size_t i=0;i<cases.size();++i) {
    work.push_back({i,size_t(1)<<30});
    if(phase!="quick") { work.push_back({i,size_t(2)<<30}); work.push_back({i,size_t(4)<<30}); }
  }
  std::ofstream raw(out+"/raw.csv");
  raw<<"pattern,chunks,grid,block,bytes,round,iters,ms,GB_s\n"<<std::setprecision(10);
  std::mt19937 random(0x90a);
  for(int round=0;round<rounds;++round) {
    std::shuffle(work.begin(),work.end(),random);
    for(auto w:work) {
      const auto& c=cases[w.index]; context.launch(c,w.bytes); context.launch(c,w.bytes);
      const double ms=context.measure(c,w.bytes); context.validate(c,w.bytes);
      raw<<c.pattern<<','<<c.chunks<<','<<c.grid<<",256,"<<w.bytes<<','<<round<<",12,"<<ms<<','<<w.bytes/(ms*1e6)<<'\n';
    }
    raw.flush(); std::fprintf(stderr,"round %d/%d\n",round+1,rounds);
  }
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
