// One-CTA FP16 MFMA load experiment, with independent exact host verification.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
using F16=float __attribute__((ext_vector_type(16)));
using U4=uint32_t __attribute__((ext_vector_type(4)));
#include "pipeline.hpp"
#include "verification.hpp"

template<int Mode,int Loads,bool Wait>
__global__ __launch_bounds__(256) void matmul_probe(const uint32_t* data,float* output,uint64_t* stamps,int steps,int records,int pattern,uint32_t seed_a,uint32_t seed_b) {
  const int lane=threadIdx.x%64,wave=threadIdx.x/64;
  constexpr int lane_bytes=Loads==1?16:(Loads==2?8:4);
  uintptr_t ptr,prefetch;
  if constexpr(Mode>=3) {
    extern __shared__ __attribute__((aligned(16))) uint32_t storage[];
    const int words=records*1024*kLdsStride;
    for(int i=threadIdx.x;i<words;i+=256) storage[i]=operand_word(uint32_t(i),pattern,Loads,kLdsStride,seed_a,seed_b);
    __syncthreads();
    // Shared->generic conversion preserves the 32-bit LDS byte offset in
    // the low word. Include the symbol base rather than assuming offset 0.
    ptr=uint32_t(reinterpret_cast<uintptr_t>(storage))+uint32_t(wave*records*1024*kLdsStride+lane*lane_bytes*kLdsStride);
    prefetch=ptr;
  } else {
    const auto base=reinterpret_cast<uintptr_t>(data)+size_t(wave)*size_t(records)*1024;
    ptr=base+lane*lane_bytes; prefetch=base+lane*16;
  }
  uint64_t begin,end,wall_begin,wall_end;
  F16 c={};
  pipeline<Mode,Loads,Wait>(uint32_t(ptr),uint32_t(ptr>>32),uint32_t(prefetch),uint32_t(prefetch>>32),
                           steps,records,c,begin,end,wall_begin,wall_end);
  if(lane==0) {
    stamps[4*wave]=begin; stamps[4*wave+1]=end;
    stamps[4*wave+2]=wall_begin; stamps[4*wave+3]=wall_end;
  }
  for(int i=0;i<16;++i) output[threadIdx.x*16+i]=c[i];
}
__global__ void fill_data(uint32_t* p,size_t n,int pattern,int loads,uint32_t seed_a,uint32_t seed_b) {
  const size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<n) p[t]=operand_word(uint32_t(t),pattern,loads,1,seed_a,seed_b);
}
__global__ void evict_reader(const U4* p,size_t n,uint32_t* sink) {
  const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t stride=size_t(gridDim.x)*blockDim.x;
  uint32_t x=0;
  for(size_t t=tid;t<n;t+=stride) {
    U4 v;
    asm volatile("global_load_dwordx4 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(v) : "v"(p+t) : "memory");
    x^=v[0]^v[1]^v[2]^v[3];
  }
  sink[tid]=x;
}
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
  ~Device() { if(p) (void)hipFree(p); }
  Device(const Device&)=delete;
};
using Launch=void(*)(const uint32_t*,float*,uint64_t*,int,int,int,uint32_t,uint32_t);
template<int M,int L,bool W> void launch(const uint32_t* d,float* o,uint64_t* s,int steps,int records,int pattern,uint32_t seed_a,uint32_t seed_b) {
  const size_t shared_bytes=M>=3?size_t(records)*4096*kLdsStride:0;
  matmul_probe<M,L,W><<<1,256,shared_bytes>>>(d,o,s,steps,records,pattern,seed_a,seed_b); HIP(hipGetLastError());
}
Launch select(int m,int l,int w) {
#define PICK(M,L) if(m==M && l==L) return w?launch<M,L,true>:launch<M,L,false>
  PICK(0,1); PICK(0,2); PICK(0,4);
  PICK(1,1); PICK(1,2); PICK(1,4);
  PICK(2,1); PICK(2,2); PICK(2,4);
  PICK(3,1); PICK(3,2); PICK(3,4);
  PICK(4,1); PICK(4,2); PICK(4,4);
#undef PICK
  throw std::runtime_error("unsupported mode/load selection");
}
struct SavedModule {
  hipModule_t handle=nullptr;
  ~SavedModule() { if(handle) (void)hipModuleUnload(handle); }
};
struct Events {
  hipEvent_t begin{},end{};
  Events() { HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end)); }
  ~Events() { (void)hipEventDestroy(begin); (void)hipEventDestroy(end); }
};
int main(int argc,char** argv) try {
  std::string mode="hot",pattern_name="legacy",module_path,symbol;
  int loads=2,wait=1,steps=4096,records=0,rounds=9,device=0,warmup_ms=200,evict_mib=-1,verify=1;
  uint32_t seed_a=1729,seed_b=2718;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i];
    if(key=="--help") {
      std::printf("--mode hot|l2|stream|lds-hot|lds-sweep --loads 1|2|4 --wait 0|1 --steps N --records N --rounds N --device N --warmup-ms N --evict-mib N --verify 0|1 --pattern legacy|hash --seed-a N --seed-b N --module OLD.hsaco --symbol OLD_KERNEL\nCompiled depth=%d unroll=%d phases=%d LDS stride=%d ready banks=%d; steps divisible by unroll. Verification enabled by default, exact at steps<=8192.\n",kDepth,kUnroll,kPhases,kLdsStride,kReadyBanks);
      return 0;
    }
    if(i+1==argc) throw std::runtime_error("missing value for "+key);
    std::string val=argv[++i];
    if(key=="--mode") mode=val;
    else if(key=="--loads") loads=std::stoi(val);
    else if(key=="--wait") wait=std::stoi(val);
    else if(key=="--steps") steps=std::stoi(val);
    else if(key=="--records") records=std::stoi(val);
    else if(key=="--rounds") rounds=std::stoi(val);
    else if(key=="--device") device=std::stoi(val);
    else if(key=="--warmup-ms") warmup_ms=std::stoi(val);
    else if(key=="--evict-mib") evict_mib=std::stoi(val);
    else if(key=="--verify") verify=std::stoi(val);
    else if(key=="--pattern") pattern_name=val;
    else if(key=="--seed-a") seed_a=uint32_t(std::stoul(val));
    else if(key=="--seed-b") seed_b=uint32_t(std::stoul(val));
    else if(key=="--module") module_path=val;
    else if(key=="--symbol") symbol=val;
    else throw std::runtime_error("unknown option "+key);
  }
  const int m=mode=="hot"?0:mode=="l2"?1:mode=="stream"?2:mode=="lds-hot"?3:mode=="lds-sweep"?4:-1;
  const int pattern=pattern_name=="legacy"?0:pattern_name=="hash"?1:-1;
  if(!module_path.empty() && kReadyBanks!=2) throw std::runtime_error("original module verification requires a two-ready-bank host build");
  if(module_path.empty()!=symbol.empty()) throw std::runtime_error("--module and --symbol must be supplied together");
  if(!module_path.empty() && m>=3 && pattern) throw std::runtime_error("original LDS code objects initialize legacy data only; hash LDS tests use the rebuilt kernel");
  if(pattern<0 || (verify!=0 && verify!=1) || (verify && steps>8192))
    throw std::runtime_error("pattern legacy|hash; verify 0|1; exact verification supports steps<=8192");
  if(m<0 || (loads!=1 && loads!=2 && loads!=4) || (wait!=0 && wait!=1) ||
     steps<=0 || steps>(1<<20)-32 || steps%kUnroll || rounds<1 || warmup_ms<0 || records<0 || records>(1<<20))
    throw std::runtime_error("invalid selection; steps positive, divisible by compiled unroll, <=1048544; records <=1048576");
  if(m==0) { if(records>1) throw std::runtime_error("hot mode requires records=1"); records=1; }
  if(m==1 && !records) records=256; // 1 MiB total, deliberately larger than L1.
  if(m==2 && !records) records=steps+kDepth+kReadyBanks;
  if(m==2 && records<steps+kDepth+kReadyBanks-1) throw std::runtime_error("stream needs at least steps+depth+ready_banks-1 records (no wrap)");
  if(m==3) { if(records>1) throw std::runtime_error("lds-hot requires records=1"); records=1; }
  if(m==4 && !records) records=16/kLdsStride;
  if(m>=3 && records>16/kLdsStride) throw std::runtime_error("LDS window exceeds 32 banks * 512 dwords (64 KiB)");
  if(evict_mib<0) evict_mib=m==2?256:0;
  if(m>=3 && evict_mib) throw std::runtime_error("LDS mode does not use the VRAM eviction reader");
  if(evict_mib<0 || evict_mib>1024) throw std::runtime_error("evict-mib must be 0..1024");
  HIP(hipSetDevice(device)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,device));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,device));
  char pci[64]{}; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),device));
  const size_t dwords=size_t(records)*1024*(m>=3?kLdsStride:1);
  Device<uint32_t> data(m>=3?1:dwords); Device<float> output(256*16+32); Device<uint64_t> stamps(16);
  if(m<3) { fill_data<<<(dwords+255)/256,256>>>(data.p,dwords,pattern,loads,seed_a,seed_b); HIP(hipGetLastError()); }
  const size_t trash_count=evict_mib?(size_t(evict_mib)<<20)/sizeof(U4):1;
  const int eviction_blocks=prop.multiProcessorCount*8;
  Device<U4> trash(trash_count); Device<uint32_t> sink(eviction_blocks*256);
  if(evict_mib) HIP(hipMemset(trash.p,0x3c,trash_count*sizeof(U4)));
  HIP(hipDeviceSynchronize());
  std::vector<float> expected;
  if(verify) {
    std::vector<uint32_t> input(dwords);
    for(size_t i=0;i<dwords;++i) input[i]=operand_word(uint32_t(i),pattern,loads,m>=3?kLdsStride:1,seed_a,seed_b);
    if(m<3) {
      std::vector<uint32_t> actual(dwords); HIP(hipMemcpy(actual.data(),data.p,dwords*4,hipMemcpyDeviceToHost));
      if(actual!=input) throw std::runtime_error("GPU input initialization differs from constructed host input");
    }
    expected=matrix_reference(input,m,loads,m>=3?kLdsStride:1,steps,records);
    // Check that the verifier itself accepts an exact result and rejects
    // independently injected output/guard corruption (host-only controls).
    std::vector<float> synthetic(4096+32,std::bit_cast<float>(0xffffffffu));
    std::copy(expected.begin(),expected.end(),synthetic.begin()+16);
    const auto exact_control=verify_matrix(synthetic,expected,false);
    if(exact_control.mismatches || exact_control.guard_errors) throw std::runtime_error("verifier exact-result control failed");
    synthetic[16]=expected[0]+1.0f;
    if(verify_matrix(synthetic,expected,false).mismatches!=1) throw std::runtime_error("verifier output-corruption control failed");
    synthetic[0]=0;
    if(verify_matrix(synthetic,expected,false).guard_errors!=1) throw std::runtime_error("verifier guard-corruption control failed");
    synthetic[0]=std::bit_cast<float>(0xffffffffu); synthetic.back()=0;
    if(verify_matrix(synthetic,expected,false).guard_errors!=1) throw std::runtime_error("verifier trailing-guard control failed");
    const auto range=std::minmax_element(expected.begin(),expected.end());
    const int nonzero=int(std::count_if(expected.begin(),expected.end(),[](float x){return x!=0;}));
    std::fprintf(stderr,"REFERENCE outputs=4096 nonzero=%d min=%.9g max=%.9g; exact dyadic oracle, verifier negative controls checked\n",nonzero,*range.first,*range.second);
  }
  const auto fn=select(m,loads,wait);
  SavedModule saved; hipFunction_t saved_function=nullptr;
  if(!module_path.empty()) {
    HIP(hipModuleLoad(&saved.handle,module_path.c_str()));
    HIP(hipModuleGetFunction(&saved_function,saved.handle,symbol.c_str()));
    std::fprintf(stderr,"ORIGINAL CODE OBJECT %s symbol=%s (original five-argument ABI)\n",module_path.c_str(),symbol.c_str());
  }
  auto run=[&]() {
    if(saved_function) {
      float* out=output.p+16;
      void* args[]={&data.p,&out,&stamps.p,&steps,&records};
      const unsigned shared=m>=3?unsigned(records*4096*kLdsStride):0;
      HIP(hipModuleLaunchKernel(saved_function,1,1,1,256,1,1,shared,nullptr,args,nullptr));
    } else fn(data.p,output.p+16,stamps.p,steps,records,pattern,seed_a,seed_b);
  };
  auto evict=[&]() {
    if(evict_mib) { evict_reader<<<eviction_blocks,256>>>(trash.p,trash_count,sink.p); HIP(hipGetLastError()); HIP(hipDeviceSynchronize()); }
  };
  std::fprintf(stderr,"ready_banks=%d; raw queue depth=%d; %s\n",kReadyBanks,kDepth,
    kReadyBanks==0?"direct raw-entry consumption; NO copies or extra arrival stage":"copies retained; R1 immediately feeds the next MFMA");
  const int requested_wait=loads*(kDepth-1);
  const int effective_wait=wait?(m>=3?std::min(requested_wait,14):requested_wait):-1;
  std::fprintf(stderr,"counter=%s requested_wait=%d effective_wait=%d lds_stride=%d; LDS queue priming is INSIDE core timing to avoid timer LGKM drains\n",m>=3?"lgkmcnt":"vmcnt",requested_wait,effective_wait,kLdsStride);
  std::fprintf(stderr,"verification=%d pattern=%s seed_a=%u seed_b=%u; no-wait remains UNSAFE regardless of any observed matches\n",verify,pattern_name.c_str(),seed_a,seed_b);
  std::fprintf(stderr,"FP16 PROBE device=%s HIP=%d PCI=%s arch=%s grid=1 block=256 waves=4 mode=%s loads=%d wait=%d depth=%d unroll=%d phases=%d steps=%d logical_K=%lld records_per_wave=%d active_bytes=%zu evict_mib=%d warmup_ms=%d; native O3, NUMA placement external, no model/context; result verification outside timing when enabled\n",
    prop.name,device,pci,prop.gcnArchName,mode.c_str(),loads,wait,kDepth,kUnroll,kPhases,steps,8LL*steps,records,dwords*4,evict_mib,warmup_ms);
  const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warmup_ms);
  while(std::chrono::steady_clock::now()<until) { evict(); run(); HIP(hipDeviceSynchronize()); }
  Events events;
  std::printf("round,mode,loads,wait,depth,unroll,phases,steps,logical_K,records,active_bytes,evict_mib,launch_us,cta_core_cycles,cta_cycles_per_step,mean_wave_cycles_per_step,min_wave_cycles_per_step,max_wave_cycles_per_step,body_us,core_MHz,logical_B_per_core_cycle,lds_stride,requested_wait,effective_wait");
  for(int w=0;w<4;++w) std::printf(",begin%d,end%d,wall_begin%d,wall_end%d",w,w,w,w);
  std::puts(",verified,mismatch_count,guard_errors,max_abs_error,valid,ready_banks");
  int invalid=0;
  for(int r=0;r<rounds;++r) {
    if(verify) HIP(hipMemset(output.p,0xff,(4096+32)*sizeof(float)));
    evict(); HIP(hipEventRecord(events.begin)); run(); HIP(hipEventRecord(events.end)); HIP(hipEventSynchronize(events.end));
    float ms=0; HIP(hipEventElapsedTime(&ms,events.begin,events.end));
    uint64_t st[16]; HIP(hipMemcpy(st,stamps.p,sizeof(st),hipMemcpyDeviceToHost));
    uint64_t first=std::numeric_limits<uint64_t>::max(),last=0,wfirst=first,wlast=0;
    double sum=0,mn=1e300,mx=0;
    for(int w=0;w<4;++w) {
      first=std::min(first,st[4*w]); last=std::max(last,st[4*w+1]);
      wfirst=std::min(wfirst,st[4*w+2]); wlast=std::max(wlast,st[4*w+3]);
      const double c=double(st[4*w+1]-st[4*w])/steps;
      sum+=c; mn=std::min(mn,c); mx=std::max(mx,c);
    }
    const double span=double(last-first),body_us=double(wlast-wfirst)*1000/wall_rate;
    std::printf("%d,%s,%d,%d,%d,%d,%d,%d,%lld,%d,%zu,%d,%.6f,%.0f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.9f",
      r,mode.c_str(),loads,wait,kDepth,kUnroll,kPhases,steps,8LL*steps,records,dwords*4,evict_mib,
      ms*1000,span,span/steps,sum/4,mn,mx,body_us,span/body_us,4096.0*steps/span);
    std::printf(",%d,%d,%d",kLdsStride,requested_wait,effective_wait);
    for(int w=0;w<4;++w) std::printf(",%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(st[4*w]),static_cast<unsigned long long>(st[4*w+1]),
      static_cast<unsigned long long>(st[4*w+2]),static_cast<unsigned long long>(st[4*w+3]));
    Verification checked;
    if(verify) {
      std::vector<float> values(4096+32); HIP(hipMemcpy(values.data(),output.p,values.size()*sizeof(float),hipMemcpyDeviceToHost));
      checked=verify_matrix(values,expected,invalid==0);
      if(checked.mismatches || checked.guard_errors) ++invalid;
    }
    std::printf(",%d,%d,%d,%.9g,%d,%d\n",verify,checked.mismatches,checked.guard_errors,checked.max_error,
                verify?int(!checked.mismatches && !checked.guard_errors):-1,kReadyBanks);
    std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE verified=%d rounds=%d invalid_samples=%d wait=%d pattern=%s\n",verify,rounds,invalid,wait,pattern_name.c_str());
  if(invalid) return 2;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
