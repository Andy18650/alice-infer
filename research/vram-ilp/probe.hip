// Fixed 104x256 streaming reads: source ILP vs generated load scheduling.
// Native gfx90a HIP/C++20; no production engine objects.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)

__global__ void initialize(uint32_t* p,size_t n) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
    p[i]=uint32_t(i)*1664525u+1013904223u;
}

// P: 0 whole-grid interleaved, 1 wave tiles, 2 CTA tiles, 3 CTA slab.
// Policy: 0 load+accumulate, 1 staged C++, 2 staged + backend sched barrier.
// The sched barrier emits no GPU fence/barrier instruction and synchronizes
// neither lanes nor CTAs. Its purpose is constraining compiler scheduling.
template<int P,int U,int Policy>
__device__ __forceinline__ void read_body(const uint2* __restrict__ p,size_t nvec,
                                         uint32_t* __restrict__ sink) {
  const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t threads=size_t(gridDim.x)*blockDim.x;
  const size_t begin=P==3?nvec*blockIdx.x/gridDim.x:0;
  const size_t end=P==3?nvec*(blockIdx.x+1)/gridDim.x:nvec;
  const size_t intra=P==0?threads:P==1?64:blockDim.x;
  const size_t next=P==3?U*blockDim.x:U*threads;
  size_t i;
  if constexpr(P==0) i=tid;
  else if constexpr(P==1) i=(tid/64)*64*U+(threadIdx.x&63);
  else if constexpr(P==2) i=size_t(blockIdx.x)*blockDim.x*U+threadIdx.x;
  else i=begin+threadIdx.x;
  uint32_t a[U][2]={};
  for(;i+(U-1)*intra<end;i+=next) {
    if constexpr(Policy==0) {
#pragma unroll
      for(int u=0;u<U;++u) { auto v=p[i+u*intra]; a[u][0]+=v.x; a[u][1]+=v.y; }
    } else {
      uint2 v[U];
#pragma unroll
      for(int u=0;u<U;++u) v[u]=p[i+u*intra];
      if constexpr(Policy==2) __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for(int u=0;u<U;++u) { a[u][0]+=v[u].x; a[u][1]+=v[u].y; }
      if constexpr(Policy==2) __builtin_amdgcn_sched_barrier(0);
    }
  }
  if constexpr(P==0 || P==3) {
    for(;i<end;i+=intra) { auto v=p[i]; a[0][0]+=v.x; a[0][1]+=v.y; }
  } else {
    // Remaining partial tile: never step into another wave's/CTA's tile.
#pragma unroll
    for(int u=0;u<U;++u) if(i+u*intra<end) {
      auto v=p[i+u*intra]; a[0][0]+=v.x; a[0][1]+=v.y;
    }
  }
  uint32_t sum=0;
#pragma unroll
  for(int u=0;u<U;++u)
#pragma unroll
    for(int w=0;w<2;++w) sum+=a[u][w];
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if((threadIdx.x&63)==0) sink[tid/64]=sum;
}

template<int P,int U,int Policy>
__global__ void read_ilp(const uint2* __restrict__ p,size_t nvec,uint32_t* __restrict__ sink) {
  read_body<P,U,Policy>(p,nvec,sink);
}
// Explicitly target the requested 256-thread CTA and permit register use
// appropriate to one CTA/CU. This is a separately labeled codegen axis.
template<int P,int U,int Policy>
__global__ __launch_bounds__(256,1)
void read_bounded(const uint2* __restrict__ p,size_t nvec,uint32_t* __restrict__ sink) {
  read_body<P,U,Policy>(p,nvec,sink);
}

// Software pipeline: keep the current tile's results while issuing the next
// tile's loads. Unlike a batch-and-drain loop, steady-state waits may leave
// the next tile outstanding. The barrier is compiler scheduling only.
template<int P,int U>
__global__ __launch_bounds__(256,1)
void read_pipelined(const uint2* __restrict__ p,size_t nvec,uint32_t* __restrict__ sink) {
  const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t threads=size_t(gridDim.x)*blockDim.x;
  const size_t begin=P==3?nvec*blockIdx.x/gridDim.x:0;
  const size_t end=P==3?nvec*(blockIdx.x+1)/gridDim.x:nvec;
  const size_t intra=P==0?threads:P==1?64:blockDim.x;
  const size_t step=P==3?U*blockDim.x:U*threads;
  size_t i;
  if constexpr(P==0) i=tid;
  else if constexpr(P==1) i=(tid/64)*64*U+(threadIdx.x&63);
  else if constexpr(P==2) i=size_t(blockIdx.x)*blockDim.x*U+threadIdx.x;
  else i=begin+threadIdx.x;
  uint32_t a[U][2]={};
  if(i+(U-1)*intra<end) {
    uint2 current[U];
#pragma unroll
    for(int u=0;u<U;++u) current[u]=p[i+u*intra];
    __builtin_amdgcn_sched_barrier(0);
    i+=step;
    for(;i+(U-1)*intra<end;i+=step) {
      uint2 next[U];
#pragma unroll
      for(int u=0;u<U;++u) next[u]=p[i+u*intra];
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for(int u=0;u<U;++u) {
        a[u][0]+=current[u].x; a[u][1]+=current[u].y;
        current[u]=next[u];
      }
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for(int u=0;u<U;++u) { a[u][0]+=current[u].x; a[u][1]+=current[u].y; }
  }
  if constexpr(P==0 || P==3) {
    for(;i<end;i+=intra) { auto v=p[i]; a[0][0]+=v.x; a[0][1]+=v.y; }
  } else {
#pragma unroll
    for(int u=0;u<U;++u) if(i+u*intra<end) {
      auto v=p[i+u*intra]; a[0][0]+=v.x; a[0][1]+=v.y;
    }
  }
  uint32_t sum=0;
#pragma unroll
  for(int u=0;u<U;++u)
#pragma unroll
    for(int w=0;w<2;++w) sum+=a[u][w];
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if((threadIdx.x&63)==0) sink[tid/64]=sum;
}

// Two explicit alternating register banks. Consume A while B is outstanding,
// refill A, then consume B while A is outstanding, and refill B. This avoids
// the current=next SSA copies that can drain the one-step pipeline's loads.
template<int P,int U>
__global__ __launch_bounds__(256,1)
void read_pingpong(const uint2* __restrict__ p,size_t nvec,uint32_t* __restrict__ sink) {
  const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t threads=size_t(gridDim.x)*blockDim.x;
  const size_t intra=P==0?threads:64;
  const size_t step=U*threads;
  size_t i=P==0?tid:(tid/64)*64*U+(threadIdx.x&63);
  uint32_t a[U][2]={};
  if(i+step+(U-1)*intra<nvec) {
    uint2 A[U],B[U];
#pragma unroll
    for(int u=0;u<U;++u) A[u]=p[i+u*intra];
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for(int u=0;u<U;++u) B[u]=p[i+step+u*intra];
    __builtin_amdgcn_sched_barrier(0);
    i+=2*step;
    for(;i+step+(U-1)*intra<nvec;i+=2*step) {
#pragma unroll
      for(int u=0;u<U;++u) { a[u][0]+=A[u].x; a[u][1]+=A[u].y; }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for(int u=0;u<U;++u) A[u]=p[i+u*intra];
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for(int u=0;u<U;++u) { a[u][0]+=B[u].x; a[u][1]+=B[u].y; }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for(int u=0;u<U;++u) B[u]=p[i+step+u*intra];
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for(int u=0;u<U;++u) { a[u][0]+=A[u].x+B[u].x; a[u][1]+=A[u].y+B[u].y; }
  }
  for(;i+(U-1)*intra<nvec;i+=step) {
#pragma unroll
    for(int u=0;u<U;++u) { auto v=p[i+u*intra]; a[u][0]+=v.x; a[u][1]+=v.y; }
  }
  if constexpr(P==0) {
    for(;i<nvec;i+=intra) { auto v=p[i]; a[0][0]+=v.x; a[0][1]+=v.y; }
  } else {
#pragma unroll
    for(int u=0;u<U;++u) if(i+u*intra<nvec) {
      auto v=p[i+u*intra]; a[0][0]+=v.x; a[0][1]+=v.y;
    }
  }
  uint32_t sum=0;
#pragma unroll
  for(int u=0;u<U;++u) sum+=a[u][0]+a[u][1];
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if((threadIdx.x&63)==0) sink[tid/64]=sum;
}

struct Stamp { uint64_t start,end; uint32_t cu,pad; };
__global__ __launch_bounds__(256,1)
void read_marked(const uint2* p,size_t nvec,uint32_t* sink,Stamp* records) {
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const uint64_t start=wall_clock64();
  if((threadIdx.x&63)==0) { records[wave].start=start; records[wave].cu=__smid(); }
  read_body<1,8,2>(p,nvec,sink);
  const uint64_t end=wall_clock64();
  if((threadIdx.x&63)==0) records[wave].end=end;
}

struct Case {
  std::string name;
  int pattern,unroll,policy,grid;
  const void* native=nullptr;
  hipFunction_t original=nullptr;
  int regs=0,active=0;
  size_t shared=0,local=0;
  int block=256;
  bool stamped=false;
};
const char* pattern_name(int p) { return p==0?"global":p==1?"wave":p==2?"CTA":"slab"; }
const char* policy_name(int p) { return p==0?"immediate":p==1?"staged_cpp":p==2?"staged_sched":p==3?"pipeline":"pingpong"; }
template<int P,int U,int Policy,bool Bounded=false>
void add(std::vector<Case>& cases,int grid) {
  const void* fn;
  if constexpr(Bounded) fn=reinterpret_cast<const void*>(read_bounded<P,U,Policy>);
  else fn=reinterpret_cast<const void*>(read_ilp<P,U,Policy>);
  hipFuncAttributes attr{}; HIP(hipFuncGetAttributes(&attr,fn));
  int active; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,0));
  cases.push_back({Bounded?"bounded":"native",P,U,Policy,grid,fn,nullptr,attr.numRegs,active,attr.sharedSizeBytes,attr.localSizeBytes});
}
template<int P,int U,bool Bounded=false> void policies(std::vector<Case>& c,int grid) {
  add<P,U,0,Bounded>(c,grid); add<P,U,1,Bounded>(c,grid); add<P,U,2,Bounded>(c,grid);
}
template<int P,bool Bounded=false> void widths(std::vector<Case>& c,int grid) {
  policies<P,1,Bounded>(c,grid); policies<P,4,Bounded>(c,grid); policies<P,8,Bounded>(c,grid);
  policies<P,16,Bounded>(c,grid); policies<P,32,Bounded>(c,grid); policies<P,64,Bounded>(c,grid);
}
template<int P,int U>
void add_pipeline(std::vector<Case>& cases,int grid) {
  const void* fn=reinterpret_cast<const void*>(read_pipelined<P,U>);
  hipFuncAttributes attr{}; HIP(hipFuncGetAttributes(&attr,fn));
  int active; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,0));
  cases.push_back({"bounded",P,U,3,grid,fn,nullptr,attr.numRegs,active,attr.sharedSizeBytes,attr.localSizeBytes});
}
template<int P> void pipelines(std::vector<Case>& c,int grid) {
  add_pipeline<P,4>(c,grid); add_pipeline<P,8>(c,grid);
  add_pipeline<P,16>(c,grid); add_pipeline<P,32>(c,grid);
}
template<int P,int U>
void add_pingpong(std::vector<Case>& cases,int grid) {
  const void* fn=reinterpret_cast<const void*>(read_pingpong<P,U>);
  hipFuncAttributes attr{}; HIP(hipFuncGetAttributes(&attr,fn));
  int active; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,0));
  cases.push_back({"bounded",P,U,4,grid,fn,nullptr,attr.numRegs,active,attr.sharedSizeBytes,attr.localSizeBytes});
}
void add_marked(std::vector<Case>& cases) {
  const void* fn=reinterpret_cast<const void*>(read_marked);
  hipFuncAttributes attr{}; HIP(hipFuncGetAttributes(&attr,fn));
  int active; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,0));
  cases.push_back({"marked",1,8,2,104,fn,nullptr,attr.numRegs,active,attr.sharedSizeBytes,attr.localSizeBytes,256,true});
}
void fields(std::ostream& o,const Case& c) {
  o<<c.name<<','<<pattern_name(c.pattern)<<','<<c.unroll<<','<<policy_name(c.policy)<<','<<c.grid<<','<<c.block;
}

struct Context {
  hipDeviceProp_t prop{};
  hipStream_t stream{}; hipEvent_t start{},end{};
  hipModule_t module{};
  uint32_t *p{},*sink{};
  Stamp* marks{};
  Context(const std::string& object,const std::string& out) {
    HIP(hipSetDevice(0)); HIP(hipGetDeviceProperties(&prop,0));
    if(prop.warpSize!=64 || prop.multiProcessorCount!=104) throw std::runtime_error("study requires 104-CU wave64 device");
    HIP(hipStreamCreate(&stream)); HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
    HIP(hipModuleLoad(&module,object.c_str()));
    HIP(hipMalloc(&p,(size_t(4)<<30)+65536)); HIP(hipMalloc(&sink,832*8*4)); HIP(hipMalloc(&marks,832*8*sizeof(Stamp)));
    initialize<<<832,256,0,stream>>>(p,((size_t(4)<<30)+65536)/4);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    int runtime; HIP(hipRuntimeGetVersion(&runtime));
    std::ofstream meta(out+"/device.txt");
    meta<<"device=0 PCI="<<pci<<" name="<<prop.name<<" arch="<<prop.gcnArchName<<" CUs="<<prop.multiProcessorCount
        <<" wave="<<prop.warpSize<<" runtime="<<runtime<<" allocation="<<static_cast<void*>(p)<<'\n';
  }
  ~Context() {
    (void)hipFree(marks); (void)hipFree(sink); (void)hipFree(p); (void)hipModuleUnload(module);
    (void)hipEventDestroy(start); (void)hipEventDestroy(end); (void)hipStreamDestroy(stream);
  }
  void add_original(std::vector<Case>& cases,int unroll,int grid,int block=256) {
    const auto symbol="_Z11read_streamILi2ELi"+std::to_string(unroll)+"ELb0EEvPKN6VectorIXT_EE4TypeEmPj";
    hipFunction_t fn; HIP(hipModuleGetFunction(&fn,module,symbol.c_str()));
    int regs,local,shared,active;
    HIP(hipFuncGetAttribute(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,fn));
    HIP(hipFuncGetAttribute(&local,HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fn));
    HIP(hipFuncGetAttribute(&shared,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn));
    HIP(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,block,0));
    cases.push_back({"preserved",0,unroll,0,grid,nullptr,fn,regs,active,size_t(shared),size_t(local),block});
  }
  void launch(const Case& c,size_t bytes,size_t offset=0) {
    const uint2* input=reinterpret_cast<const uint2*>(p+offset/4);
    size_t nvec=bytes/8;
    void* args[]={&input,&nvec,&sink};
    if(c.stamped) {
      void* marked_args[]={&input,&nvec,&sink,&marks};
      HIP(hipLaunchKernel(c.native,dim3(c.grid),dim3(c.block),marked_args,0,stream));
    } else if(c.original) HIP(hipModuleLaunchKernel(c.original,c.grid,1,1,c.block,1,1,0,stream,args,nullptr));
    else HIP(hipLaunchKernel(c.native,dim3(c.grid),dim3(c.block),args,0,stream));
  }
  void validate(const Case& c,size_t bytes,size_t offset=0) {
    std::vector<uint32_t> h(c.grid*(c.block/64));
    HIP(hipMemcpyAsync(h.data(),sink,h.size()*4,hipMemcpyDeviceToHost,stream)); HIP(hipStreamSynchronize(stream));
    uint32_t sum=0; for(auto v:h) sum+=v;
    uint64_t n=bytes/4;
    const uint32_t expected=uint32_t(n*(n-1)/2+n*(offset/4))*1664525u+uint32_t(n)*1013904223u;
    if(sum!=expected) throw std::runtime_error("checksum mismatch: "+c.name+" "+pattern_name(c.pattern)+" U="+std::to_string(c.unroll)+" "+policy_name(c.policy));
  }
  double measure(const Case& c,size_t bytes,int iters,size_t offset=0) {
    HIP(hipEventRecord(start,stream));
    for(int i=0;i<iters;++i) launch(c,bytes,offset);
    HIP(hipGetLastError()); HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
    float ms; HIP(hipEventElapsedTime(&ms,start,end)); return double(ms)/iters;
  }
};

void trace(Context& context,const std::string& out) {
  Stamp* d; HIP(hipMalloc(&d,104*4*sizeof(Stamp)));
  size_t nvec=(size_t(1)<<30)/8;
  const uint2* input=reinterpret_cast<const uint2*>(context.p);
  void* args[]={&input,&nvec,&context.sink,&d};
  const void* fn=reinterpret_cast<const void*>(read_marked);
  hipFuncAttributes attr{}; HIP(hipFuncGetAttributes(&attr,fn));
  int active,rate; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,0));
  HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));
  if(rate<=0 || attr.localSizeBytes) throw std::runtime_error("invalid clock or scratch-using tracer");
  std::ofstream summary(out+"/trace-summary.csv");
  summary<<"sample,reported_regs,capacity_CTAs_per_CU,CUs_seen,max_CTAs_assigned_one_CU,wave_entry_spread_us,all_wave_common_interval_us\n";
  for(int i=0;i<20;++i) HIP(hipLaunchKernel(fn,dim3(104),dim3(256),args,0,context.stream));
  for(int sample=0;sample<7;++sample) {
    HIP(hipMemsetAsync(d,0,104*4*sizeof(Stamp),context.stream));
    HIP(hipLaunchKernel(fn,dim3(104),dim3(256),args,0,context.stream));
    const Case check{"marked",1,8,2,104}; context.validate(check,size_t(1)<<30);
    std::vector<Stamp> records(104*4);
    HIP(hipMemcpyAsync(records.data(),d,records.size()*sizeof(Stamp),hipMemcpyDeviceToHost,context.stream));
    HIP(hipStreamSynchronize(context.stream));
    std::vector<std::pair<uint32_t,int>> assignment;
    uint64_t first=UINT64_MAX,last=0,first_end=UINT64_MAX;
    std::ofstream raw(out+"/trace-"+std::to_string(sample)+".csv");
    raw<<"CTA,wave,CU,start_tick,end_tick\n";
    for(int cta=0;cta<104;++cta) {
      const auto cu=records[cta*4].cu;
      auto it=std::find_if(assignment.begin(),assignment.end(),[&](auto v){return v.first==cu;});
      if(it==assignment.end()) assignment.push_back({cu,1}); else ++it->second;
      for(int w=0;w<4;++w) {
        const auto& r=records[cta*4+w];
        if(r.end<=r.start || r.cu!=cu) throw std::runtime_error("invalid trace");
        first=std::min(first,r.start); last=std::max(last,r.start); first_end=std::min(first_end,r.end);
        raw<<cta<<','<<w<<','<<cu<<','<<r.start<<','<<r.end<<'\n';
      }
    }
    int max_assigned=0; for(auto [cu,count]:assignment) { (void)cu; max_assigned=std::max(max_assigned,count); }
    summary<<sample<<','<<attr.numRegs<<','<<active<<','<<assignment.size()<<','<<max_assigned<<','
           <<double(last-first)*1000/rate<<','<<(first_end>last?double(first_end-last)*1000/rate:0)<<'\n';
  }
  HIP(hipFree(d));
}

int main(int argc,char** argv) try {
  if(argc<3 || argc>5) throw std::runtime_error("usage: probe OUT PRESERVED_CODE_OBJECT [ROUNDS=9] [all|focus|marked|trace]");
  const std::string out=argv[1]; const int rounds=argc>3?std::stoi(argv[3]):9;
  const std::string phase=argc>4?argv[4]:"all";
  if(phase!="all" && phase!="focus" && phase!="marked" && phase!="trace") throw std::runtime_error("phase: all|focus|marked|trace");
  if(rounds<3 || rounds%2==0) throw std::runtime_error("odd rounds >=3");
  std::filesystem::create_directories(out);
  Context context(argv[2],out);
  if(phase=="trace") { trace(context,out); return 0; }
  std::vector<Case> cases;
  widths<0>(cases,104); widths<1>(cases,104); widths<2>(cases,104); widths<3>(cases,104);
  widths<0,true>(cases,104); widths<1,true>(cases,104); widths<2,true>(cases,104); widths<3,true>(cases,104);
  pipelines<0>(cases,104); pipelines<1>(cases,104); pipelines<2>(cases,104); pipelines<3>(cases,104);
  add_pingpong<0,4>(cases,104); add_pingpong<0,8>(cases,104); add_pingpong<0,16>(cases,104);
  add_pingpong<1,4>(cases,104); add_pingpong<1,8>(cases,104); add_pingpong<1,16>(cases,104);
  for(int u:{1,4,8,16}) context.add_original(cases,u,104);
  for(int u:{8,16}) context.add_original(cases,u,832);
  context.add_original(cases,16,832,128);
  context.add_original(cases,16,1664,128);
  context.add_original(cases,1,832,512);
  add<0,8,0>(cases,832); add<0,16,0>(cases,832); add<0,16,2>(cases,832);
  add_marked(cases);
  std::ofstream resources(out+"/resources.csv");
  resources<<"source,pattern,unroll,policy,grid,block,reported_regs,shared_bytes,private_bytes,capacity_CTAs_per_CU,status\n";
  for(const auto& c:cases) {
    fields(resources,c); resources<<','<<c.regs<<','<<c.shared<<','<<c.local<<','<<c.active<<','<<(c.local?"excluded_scratch":"eligible")<<'\n';
  }
  cases.erase(std::remove_if(cases.begin(),cases.end(),[](const Case& c){return c.local!=0;}),cases.end());
  if(phase=="focus") cases.erase(std::remove_if(cases.begin(),cases.end(),[](const Case& c) {
    if(c.name=="preserved") return c.unroll==4 || (c.grid==832 && c.unroll==8);
    if(c.grid!=104) return true;
    if(c.policy>=3) return !(c.unroll==8 || (c.pattern<=1 && c.unroll==16) || (c.pattern==1 && c.unroll==4));
    if(c.name=="native") return !((c.pattern==0 && c.unroll==16 && c.policy!=1)
      || (c.pattern==1 && c.unroll==8 && c.policy!=1) || (c.pattern==2 && c.unroll==16 && c.policy==0));
    return !((c.pattern==1 && c.unroll==8 && c.policy==2) || (c.pattern==2 && c.unroll==64 && c.policy==0));
  }),cases.end());
  if(phase=="marked") cases.erase(std::remove_if(cases.begin(),cases.end(),[](const Case& c) {
    if(c.name=="marked") return false;
    if(c.name=="preserved") return !((c.grid==104 && c.unroll==8) || (c.grid==832 && c.block==256 && c.unroll==16));
    if(c.grid!=104) return true;
    return !((c.name=="native" && ((c.pattern==1 && c.unroll==8 && c.policy==2) || (c.pattern==0 && c.unroll==16 && c.policy==2)))
      || (c.name=="bounded" && c.pattern==1 && c.unroll==8 && c.policy==3));
  }),cases.end());
  for(const auto& c:cases) for(size_t bytes:{size_t(64),size_t(4160),size_t((1<<20)+64)}) {
    context.launch(c,bytes); HIP(hipGetLastError()); context.validate(c,bytes);
  }
  std::fprintf(stderr,"%zu non-scratch cases passed small/tail checks\n",cases.size());
  const Case& warm=cases.front();
  for(int i=0;i<250;++i) context.launch(warm,size_t(1)<<30);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(context.stream));
  struct Work { size_t case_index,bytes,offset; };
  std::vector<Work> work;
  const std::vector<size_t> offsets=(phase=="focus" || phase=="marked")?std::vector<size_t>{0,256,8192,65536}:std::vector<size_t>{0};
  for(size_t i=0;i<cases.size();++i) for(size_t bytes:{size_t(1)<<30,size_t(2)<<30,size_t(4)<<30})
    for(size_t offset:offsets) work.push_back({i,bytes,offset});
  std::mt19937 random(0x90a);
  std::ofstream raw(out+"/raw.csv");
  raw<<"source,pattern,unroll,policy,grid,block,bytes,offset,round,iters,ms,GB_s\n"<<std::setprecision(10);
  for(int round=0;round<rounds;++round) {
    std::shuffle(work.begin(),work.end(),random);
    for(const auto& w:work) {
      const auto& c=cases[w.case_index];
      context.launch(c,w.bytes,w.offset); context.launch(c,w.bytes,w.offset);
      const double ms=context.measure(c,w.bytes,12,w.offset);
      context.validate(c,w.bytes,w.offset);
      fields(raw,c); raw<<','<<w.bytes<<','<<w.offset<<','<<round<<",12,"<<ms<<','<<w.bytes/(ms*1e6)<<'\n';
    }
    raw.flush(); std::fprintf(stderr,"round %d/%d complete\n",round+1,rounds);
  }
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
