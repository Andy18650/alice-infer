// Address-pattern contention probe, native HIP/C++20; no engine dependency.
// Measures useful bytes, NOT physical HBM transactions or channel identity.
#include <hip/hip_runtime.h>
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
constexpr size_t GiB=size_t(1)<<30;
constexpr uint32_t multiplier=1664525u, increment=1013904223u, upper_mix=0x9e3779b9u;

__global__ void initialize(uint32_t* p, size_t n) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
    p[i]=uint32_t(i)*multiplier+increment+uint32_t(i>>32)*upper_mix;
    // Include high bits: a 16 GiB plane shift must not repeat input values.
}

// One wave per tile, grid-stride over tiles. For 64/128 B tiles, only
// 16/32 lanes load; 256 B uses 64 x uint32, 512 B uses 64 x uint2.
// Each wave loads all streams, with separate accumulators per stream and
// four independent tiles in flight at source level. No shared input lines
// between streams: bases separated by a configured plane gap (default
// 8 GiB), with bounds and nonoverlapping spans checked on host.
template<int Tile, int Streams, int U>
__global__ void read_tiles(const uint32_t* __restrict__ p, size_t tiles,
                           size_t stride_words, size_t base_words,
                           size_t gap_words, uint32_t* __restrict__ sink) {
  const unsigned lane=threadIdx.x&63;
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const size_t waves=size_t(gridDim.x)*blockDim.x/64;
  uint32_t a[Streams][U][2]={};
  const bool active=Tile>=256 || lane<Tile/4;
  size_t t=wave;
  for(;t+(U-1)*waves<tiles;t+=U*waves) {
#pragma unroll
    for(int u=0;u<U;++u) {
#pragma unroll
      for(int s=0;s<Streams;++s) {
        const size_t index=base_words+size_t(s)*gap_words+(t+u*waves)*stride_words;
        if(active) {
          if constexpr(Tile==512) {
            const uint2 v=reinterpret_cast<const uint2*>(p+index)[lane];
            a[s][u][0]+=v.x; a[s][u][1]+=v.y;
          } else a[s][u][0]+=p[index+lane];
        }
      }
    }
  }
  for(;t<tiles;t+=waves) {
#pragma unroll
    for(int s=0;s<Streams;++s) {
      const size_t index=base_words+size_t(s)*gap_words+t*stride_words;
      if(active) {
        if constexpr(Tile==512) {
          const uint2 v=reinterpret_cast<const uint2*>(p+index)[lane];
          a[s][0][0]+=v.x; a[s][0][1]+=v.y;
        } else a[s][0][0]+=p[index+lane];
      }
    }
  }
  uint32_t sum=0;
#pragma unroll
  for(int s=0;s<Streams;++s)
#pragma unroll
    for(int u=0;u<U;++u) sum+=a[s][u][0]+a[s][u][1];
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if(lane==0) sink[wave]=sum;
}

struct Case {
  int tile=256, streams=1, unroll=4, block=256, grid=832;
  size_t stride=8192, base=0, delta=0;
  std::string group;
  size_t plane=0;  // zero uses the process-wide default; planes phase overrides
};
void fields(std::ostream& o,const Case& c) {
  o<<c.group<<','<<c.tile<<','<<c.streams<<','<<c.unroll<<','<<c.block<<','<<c.grid
   <<','<<c.stride<<','<<c.base<<','<<c.delta;
}

struct Probe {
  hipDeviceProp_t prop{};
  hipStream_t stream{};
  hipEvent_t start{},end{};
  uint32_t *p{},*sink{};
  const size_t allocation=36*GiB;
  size_t plane_gap;
  Probe(int device,const std::string& out,size_t gap):plane_gap(gap) {
    HIP(hipSetDevice(device)); HIP(hipGetDeviceProperties(&prop,device));
    if(prop.warpSize!=64) throw std::runtime_error("requires wave64");
    size_t free,total; HIP(hipMemGetInfo(&free,&total));
    if(free<allocation+GiB) throw std::runtime_error("need 37 GiB free VRAM");
    HIP(hipStreamCreate(&stream)); HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
    HIP(hipMalloc(&p,allocation)); HIP(hipMalloc(&sink,prop.multiProcessorCount*32*4));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),device));
    int runtime; HIP(hipRuntimeGetVersion(&runtime));
    std::ofstream meta(out+"/device.txt");
    meta<<"device="<<device<<" PCI="<<pci<<" name="<<prop.name<<" arch="<<prop.gcnArchName
        <<" CUs="<<prop.multiProcessorCount<<" wave="<<prop.warpSize<<" runtime="<<runtime
        <<" allocation="<<allocation<<" plane_gap="<<plane_gap<<" pointer="<<static_cast<void*>(p)<<'\n';
    initialize<<<prop.multiProcessorCount*8,256,0,stream>>>(p,allocation/4);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  }
  ~Probe() {
    (void)hipFree(sink); (void)hipFree(p); (void)hipEventDestroy(start);
    (void)hipEventDestroy(end); (void)hipStreamDestroy(stream);
  }
  size_t tiles(const Case& c,size_t bytes) const {
    if(bytes%(c.tile*c.streams)) throw std::runtime_error("nonintegral tiles");
    return bytes/(c.tile*c.streams);
  }
  size_t plane(const Case& c) const { return c.plane?c.plane:plane_gap; }
  void launch(const Case& c,size_t bytes) {
    const size_t n=tiles(c,bytes),gap=c.streams==1?0:plane(c)+c.delta;
    const size_t span=(n-1)*c.stride+c.tile;
    if(c.stride<size_t(c.tile) || c.base%8 || c.stride%8 || gap%8
       || c.base+(c.streams-1)*gap+span>allocation
       || (c.streams>1 && span>gap)) throw std::runtime_error("invalid/overlapping/out-of-bounds case");
#define RUN(T,S,U) if(c.tile==T && c.streams==S && c.unroll==U) { \
  read_tiles<T,S,U><<<c.grid,c.block,0,stream>>>(p,n,c.stride/4,c.base/4,gap/4,sink); return; }
    RUN(64,1,4) RUN(128,1,4) RUN(256,1,4) RUN(512,1,4)
    RUN(256,2,4) RUN(256,4,4)
    RUN(256,1,1) RUN(256,1,16) RUN(256,2,1) RUN(256,2,16)
#undef RUN
    throw std::runtime_error("unsupported specialization");
  }
  void validate(const Case& c,size_t bytes) {
    const uint64_t n=tiles(c,bytes),words=c.tile/4;
    const uint64_t gap=c.streams==1?0:(plane(c)+c.delta)/4;
    uint32_t expected=0;
    for(int s=0;s<c.streams;++s) {
      const uint64_t b=c.base/4+s*gap;
      const uint64_t sum_indices=n*words*b+words*(c.stride/4)*(n*(n-1)/2)
                               +n*(words*(words-1)/2);
      // Sum floor(word_index/2^32) analytically. Allocation <48 GiB,
      // so only two thresholds can be crossed; tiles do not overlap.
      uint64_t sum_upper=0;
      for(uint64_t threshold:{uint64_t(1)<<32,uint64_t(2)<<32}) {
        const uint64_t step=c.stride/4;
        const uint64_t full=threshold<=b?0:(threshold-b+step-1)/step;
        if(full<n) sum_upper+=(n-full)*words;
        if(full>0 && full-1<n) {
          const uint64_t last=b+(full-1)*step;
          if(last+words>threshold) sum_upper+=last+words-threshold;
        }
      }
      expected+=uint32_t(sum_indices)*multiplier+uint32_t(n*words)*increment
                +uint32_t(sum_upper)*upper_mix;
    }
    std::vector<uint32_t> h(size_t(c.grid)*c.block/64);
    HIP(hipMemcpyAsync(h.data(),sink,h.size()*4,hipMemcpyDeviceToHost,stream));
    HIP(hipStreamSynchronize(stream));
    uint32_t sum=0; for(auto v:h) sum+=v;
    if(sum!=expected) throw std::runtime_error("checksum mismatch");
  }
  double measure(const Case& c,size_t bytes,int iters) {
    HIP(hipEventRecord(start,stream));
    for(int i=0;i<iters;++i) launch(c,bytes);
    HIP(hipGetLastError()); HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
    float ms; HIP(hipEventElapsedTime(&ms,start,end));
    return ms/iters;
  }
};

std::vector<Case> make_cases(const std::string& phase,int cus) {
  std::vector<Case> cases;
  auto add=[&](int tile,int streams,size_t stride,size_t base,size_t delta,bool high,
               const std::string& group,int unroll=4) {
    if(stride<size_t(tile)) return;
    cases.push_back({tile,streams,unroll,high?256:64,cus*(high?8:1),stride,base,delta,group});
  };
  if(phase=="stride") {
    const std::vector<size_t> strides={64,128,256,512,768,1024,1280,1792,2048,2304,
      3840,4096,4352,7680,7936,8064,8128,8192,8256,8320,8448,8704,12288,
      15872,16128,16256,16320,16384,16448,16512,16640,16896,
      32256,32512,32768,33024,65024,65280,65536};
    for(int tile:{64,128,256,512}) for(size_t stride:strides)
      for(size_t base:{size_t(0),size_t(256)}) for(bool high:{false,true})
        add(tile,1,stride,base,0,high,"stride");
  } else if(phase=="offset") {
    for(size_t stride:{8192u,8448u,16384u,16640u,32768u,33024u})
      for(size_t base=0;base<=1024;base+=32) for(bool high:{false,true})
        add(256,1,stride,base,0,high,"offset");
  } else if(phase=="bits") {
    for(size_t stride:{8192u,8448u,16384u,16640u}) for(bool high:{false,true}) {
      for(size_t base:{size_t(0),size_t(256)}) add(256,1,stride,base,0,high,"bits");
      for(int b=12;b<=30;++b) for(size_t low:{size_t(0),size_t(256)})
        add(256,1,stride,(size_t(1)<<b)+low,0,high,"bits");
    }
  } else if(phase=="pair") {
    for(int streams:{2,4}) for(size_t stride:{8192u,8448u,16384u,16640u})
      for(size_t base:{size_t(0),size_t(256)}) for(size_t d=0;d<=8192;d+=256)
        for(bool high:{false,true}) add(256,streams,stride,base,d,high,"pair");
  } else if(phase=="pair-bits") {
    for(size_t stride:{8192u,8448u,16384u,16640u}) for(size_t base:{size_t(0),size_t(256)}) {
      add(256,2,stride,base,0,true,"pair-bits");
      for(int b=12;b<=30;++b) for(size_t low:{size_t(0),size_t(256)})
        add(256,2,stride,base,(size_t(1)<<b)+low,true,"pair-bits");
    }
  } else if(phase=="conflict") {
    for(size_t stride:{16384u,32768u,65536u})
      for(int streams:{2,4}) for(size_t base:{size_t(0),size_t(256)})
        for(size_t d=0;d<=2*stride;d+=stride/64)
          add(256,streams,stride,base,d,true,"conflict");
  } else if(phase=="planes") {
    for(size_t stride:{8192u,8448u,16384u,16640u,32768u,33024u,65536u,65792u})
      for(size_t base:{size_t(0),size_t(256)}) for(size_t d:{size_t(0),stride/2})
        for(int u:{1,4,16}) for(size_t plane:{8*GiB,16*GiB,24*GiB}) {
          add(256,2,stride,base,d,true,"planes",u);
          cases.back().plane=plane;
        }
  } else if(phase=="profile") {
    for(size_t stride:{8192u,8448u,16384u,16640u,32768u,33024u,65536u,65792u})
      add(256,1,stride,0,0,true,"profile");
    for(size_t stride:{16384u,32768u,65536u})
      for(size_t d:{size_t(0),stride/2,stride})
        add(256,2,stride,0,d,true,"profile-pair");
  } else if(phase=="confirm") {
    // Predeclared controls, not best-of-screen picks: low/high concurrency,
    // three load schedules, multiple bases and two useful traffic sizes.
    for(size_t stride:{8192u,8448u,16384u,16640u,32768u,33024u,65536u,65792u}) for(size_t base:{size_t(0),size_t(256),size_t(1)<<20,size_t(1)<<28})
      for(bool high:{false,true}) for(int u:{1,4,16})
        add(256,1,stride,base,0,high,"confirm",u);
    for(size_t stride:{8192u,8448u,16384u,16640u,32768u,33024u,65536u,65792u}) for(size_t d:{0u,256u,4096u,8192u,16384u,32768u})
      for(size_t base:{size_t(0),size_t(256)}) for(bool high:{false,true})
        add(256,2,stride,base,d,high,"confirm-pair");
  } else throw std::runtime_error("phase: stride|offset|bits|pair|pair-bits|conflict|planes|profile|confirm");
  return cases;
}

int main(int argc,char** argv) try {
  if(argc<3) throw std::runtime_error("usage: probe OUT PHASE [DEVICE=0] [PAYLOAD_MiB=32] [ROUNDS=5] [PLANE_GiB=8]");
  const std::string out=argv[1],phase=argv[2];
  const int dev=argc>3?std::stoi(argv[3]):0;
  const size_t bytes=size_t(argc>4?std::stoul(argv[4]):32)<<20;
  const int rounds=argc>5?std::stoi(argv[5]):5;
  const size_t gap=size_t(argc>6?std::stoul(argv[6]):8)*GiB;
  if(gap!=8*GiB && gap!=16*GiB) throw std::runtime_error("plane gap must be 8 or 16 GiB");
  if(bytes<16*(size_t(1)<<20)||rounds<3||rounds%2==0) throw std::runtime_error("payload >=16 MiB; odd rounds >=3");
  std::filesystem::create_directories(out);
  Probe probe(dev,out,gap);
  auto cases=make_cases(phase,probe.prop.multiProcessorCount);
  std::mt19937 random(0x90a);
  // Non-power-of-two tile counts catch missing tails/addressing mistakes.
  if(phase!="profile") for(const auto& c:cases) {
    const size_t small=size_t(c.tile)*c.streams*17;
    probe.launch(c,small); HIP(hipGetLastError()); probe.validate(c,small);
    probe.launch(c,bytes); HIP(hipGetLastError()); probe.validate(c,bytes);
  }
  std::fprintf(stderr,"%zu cases; short/full checksum controls %s\n",cases.size(),phase=="profile"?"skipped (post-read check only)":"passed");
  const Case warm{256,1,4,256,probe.prop.multiProcessorCount*8,256,0,0,"warm"};
  if(phase!="profile") for(int i=0;i<500;++i) probe.launch(warm,bytes);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(probe.stream));
  std::ofstream raw(out+"/raw.csv");
  raw<<"group,tile,streams,unroll,block,grid,stride,base,delta,bytes,span_per_stream,plane_gap,round,iters,ms,GB_s\n";
  raw<<std::setprecision(10);
  const int iters=phase=="profile"?1:phase=="confirm"?12:6;
  for(int round=0;round<rounds;++round) {
    std::shuffle(cases.begin(),cases.end(),random);
    for(const auto& c:cases) {
      if(phase!="profile") { probe.launch(c,bytes); probe.launch(c,bytes); }
      const double ms=probe.measure(c,bytes,iters);
      probe.validate(c,bytes);
      fields(raw,c); raw<<','<<bytes<<','<<(probe.tiles(c,bytes)-1)*c.stride+c.tile
          <<','<<probe.plane(c)<<','<<round<<','<<iters<<','<<ms<<','<<bytes/(ms*1e6)<<'\n';
    }
    raw.flush(); std::fprintf(stderr,"round %d/%d complete\n",round+1,rounds);
  }
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
