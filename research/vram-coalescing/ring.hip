// Rotating register-buffer reads, matched against batch A/B and the exact
// historical uint2/U=16 code object. Compare free and ordering-constrained rings.
#include "wave-control.hpp"
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
static_assert(RESEARCH_SCHED_BARRIERS==0,"ring test must not use scheduling barriers");

template<class Reader,int Depth,bool Ordered=false>
__global__ __launch_bounds__(256,1)
void read_ring(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const size_t waves=size_t(gridDim.x)*blockDim.x/64;
  const unsigned lane=threadIdx.x&63;
  const size_t full_tiles=count/256;
  uint4 sums[Depth]={};
  size_t next=wave;
  if(wave+(Depth-1)*waves<full_tiles) {
    uint4 buffer[Depth];
#pragma unroll
    for(int slot=0;slot<Depth;++slot) buffer[slot]=Reader::load(input,(wave+slot*waves)*256,lane,64);
    if constexpr(Ordered) __builtin_amdgcn_sched_barrier(0);
    next+=Depth*waves;
    for(;next+(Depth-1)*waves<full_tiles;next+=Depth*waves) {
#pragma unroll
      for(int slot=0;slot<Depth;++slot) {
        sums[slot]+=buffer[slot];
        buffer[slot]=Reader::load(input,(next+slot*waves)*256,lane,64);
        if constexpr(Ordered) __builtin_amdgcn_sched_barrier(0);
      }
    }
    // Every buffered tile was read exactly once; drain it before the tail.
#pragma unroll
    for(int slot=0;slot<Depth;++slot) sums[slot]+=buffer[slot];
  }
  // Covers short inputs, partial batches, and the final partial tile.
  for(;next*256<count;next+=waves) sums[0]+=Reader::tail(input,next*256,lane,count,64);
  uint4 total={};
#pragma unroll
  for(int slot=0;slot<Depth;++slot) total+=sums[slot];
  uint32_t sum=total.x+total.y+total.z+total.w;
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if(lane==0) sink[wave]=sum;
}

struct Case {
  std::string name;
  int chunks,grid;
  const void* native{};
  hipFunction_t legacy{};
  int registers{},capacity{};
  size_t scratch{};
};
template<class Reader,int Depth,bool Ordered=false> void add_ring(std::vector<Case>& cases) {
  const void* fn=reinterpret_cast<const void*>(read_ring<Reader,Depth,Ordered>);
  hipFuncAttributes a{}; HIP(hipFuncGetAttributes(&a,fn));
  int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,fn,256,0));
  cases.push_back({std::string(Reader::name)+(Ordered?"_ring_ordered":"_ring"),Depth,104,fn,nullptr,a.numRegs,capacity,a.localSizeBytes});
}
template<class Reader> void add_reader(std::vector<Case>& cases) {
  const void* fn=reinterpret_cast<const void*>(read_wave_tiles<Reader,4>);
  hipFuncAttributes a{}; HIP(hipFuncGetAttributes(&a,fn));
  int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,fn,256,0));
  cases.push_back({std::string(Reader::name)+"_batch",4,104,fn,nullptr,a.numRegs,capacity,a.localSizeBytes});
  add_ring<Reader,1>(cases); add_ring<Reader,2>(cases); add_ring<Reader,4>(cases);
  add_ring<Reader,8>(cases); add_ring<Reader,16>(cases);
  add_ring<Reader,2,true>(cases); add_ring<Reader,4,true>(cases);
  add_ring<Reader,8,true>(cases); add_ring<Reader,16,true>(cases);
}

struct Context {
  hipStream_t stream{}; hipEvent_t start{},end{}; hipModule_t module{};
  uint32_t *input{},*sink{};
  Context(const std::string& object,const std::string& out) {
    HIP(hipSetDevice(0));
    hipDeviceProp_t p{}; HIP(hipGetDeviceProperties(&p,0));
    if(p.warpSize!=64 || p.multiProcessorCount!=104) throw std::runtime_error("requires 104-CU wave64 device");
    HIP(hipStreamCreate(&stream)); HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
    HIP(hipModuleLoad(&module,object.c_str()));
    HIP(hipMalloc(&input,size_t(4)<<30)); HIP(hipMalloc(&sink,832*4*4));
    initialize<<<832,256,0,stream>>>(input,(size_t(4)<<30)/4);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    std::ofstream meta(out+"/device.txt");
    meta<<"device=0 PCI="<<pci<<" arch="<<p.gcnArchName<<" CUs="<<p.multiProcessorCount
        <<" input=uint32_t vector=uint4 generator=affine_mod2^32 scheduling_barriers=0\n";
  }
  ~Context() {
    (void)hipFree(sink); (void)hipFree(input); (void)hipModuleUnload(module);
    (void)hipEventDestroy(start); (void)hipEventDestroy(end); (void)hipStreamDestroy(stream);
  }
  void add_legacy(std::vector<Case>& cases) {
    hipFunction_t fn;
    HIP(hipModuleGetFunction(&fn,module,"_Z11read_streamILi2ELi16ELb0EEvPKN6VectorIXT_EE4TypeEmPj"));
    int regs,capacity,local;
    HIP(hipFuncGetAttribute(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,fn));
    HIP(hipFuncGetAttribute(&local,HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fn));
    HIP(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,fn,256,0));
    cases.push_back({"legacy_uint2_U16",16,832,nullptr,fn,regs,capacity,size_t(local)});
  }
  void launch(const Case& c,size_t bytes) {
    size_t count=bytes/(c.legacy?8:4); void* args[]={&input,&count,&sink};
    if(c.legacy) HIP(hipModuleLaunchKernel(c.legacy,c.grid,1,1,256,1,1,0,stream,args,nullptr));
    else HIP(hipLaunchKernel(c.native,dim3(c.grid),dim3(256),args,0,stream));
  }
  void validate(const Case& c,size_t bytes) {
    std::vector<uint32_t> values(c.grid*4);
    HIP(hipMemcpyAsync(values.data(),sink,values.size()*4,hipMemcpyDeviceToHost,stream)); HIP(hipStreamSynchronize(stream));
    uint32_t sum=0; for(auto v:values) sum+=v;
    if(sum!=expected_sum(bytes/4)) throw std::runtime_error("checksum mismatch: "+c.name);
  }
  double measure(const Case& c,size_t bytes) {
    HIP(hipEventRecord(start,stream)); for(int i=0;i<12;++i) launch(c,bytes);
    HIP(hipGetLastError()); HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
    float ms; HIP(hipEventElapsedTime(&ms,start,end)); return double(ms)/12;
  }
};

int main(int argc,char** argv) try {
  if(argc<3 || argc>4) throw std::runtime_error("usage: ring OUT HISTORICAL_CODE_OBJECT [ROUNDS=9]");
  const std::string out=argv[1]; const int rounds=argc>3?std::stoi(argv[3]):9;
  if(rounds<3 || rounds%2==0) throw std::runtime_error("odd rounds >=3");
  std::filesystem::create_directories(out);
  Context context(argv[2],out);
  std::vector<Case> cases; add_reader<A>(cases); add_reader<B>(cases); context.add_legacy(cases);
  std::ofstream resources(out+"/resources.csv");
  resources<<"pattern,chunks,grid,block,reported_regs,capacity_CTAs_per_CU,private_bytes,status\n";
  for(const auto& c:cases) resources<<c.name<<','<<c.chunks<<','<<c.grid<<",256,"<<c.registers<<','<<c.capacity<<','<<c.scratch<<','<<(c.scratch?"excluded_scratch":"eligible")<<'\n';
  cases.erase(std::remove_if(cases.begin(),cases.end(),[](const Case& c){return c.scratch!=0;}),cases.end());
  for(const auto& c:cases) for(size_t bytes:{size_t(64),size_t(4160),size_t((1<<20)+64)}) {
    context.launch(c,bytes); HIP(hipGetLastError()); context.validate(c,bytes);
  }
  for(int i=0;i<250;++i) context.launch(cases.front(),size_t(1)<<30);
  HIP(hipStreamSynchronize(context.stream));
  struct Work { size_t index,bytes; };
  std::vector<Work> work;
  for(size_t i=0;i<cases.size();++i) for(size_t bytes:{size_t(1)<<30,size_t(2)<<30,size_t(4)<<30}) work.push_back({i,bytes});
  std::ofstream raw(out+"/raw.csv");
  raw<<"pattern,chunks,grid,block,bytes,round,iters,ms,GB_s\n"<<std::setprecision(10);
  std::mt19937 random(0x90a);
  for(int round=0;round<rounds;++round) {
    std::shuffle(work.begin(),work.end(),random);
    for(auto w:work) {
      const auto& c=cases[w.index]; context.launch(c,w.bytes); context.launch(c,w.bytes);
      const double ms=context.measure(c,w.bytes); context.validate(c,w.bytes);
      raw<<c.name<<','<<c.chunks<<','<<c.grid<<",256,"<<w.bytes<<','<<round<<",12,"<<ms<<','<<w.bytes/(ms*1e6)<<'\n';
    }
    raw.flush(); std::fprintf(stderr,"round %d/%d complete\n",round+1,rounds);
  }
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
