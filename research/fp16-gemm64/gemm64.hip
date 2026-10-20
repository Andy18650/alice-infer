#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "data.hpp"
#define HIP(x) do {auto hip_status_=(x);if(hip_status_!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(hip_status_));}while(0)
using F16=float __attribute__((ext_vector_type(16)));
#include "compute.hpp"
__device__ __forceinline__ void begin_clock(F16& c,uint64_t& t,uint64_t& w) {
  asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)\n\ts_memrealtime %1\n\ts_waitcnt lgkmcnt(0)"
               : "=&s"(t),"=&s"(w),"+a"(c) : : "memory");
}
__device__ __forceinline__ void end_clock(F16& c,uint64_t& t,uint64_t& w) {
  asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)\n\ts_memrealtime %1\n\ts_waitcnt lgkmcnt(0)\n\t"
               "s_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_nop 15"
               : "=&s"(t),"=&s"(w),"+a"(c) : : "memory");
}
__device__ __forceinline__ void save(F16 c,float* out,uint64_t* stamps,uint64_t b,uint64_t e,uint64_t wb,uint64_t we) {
  int lane=threadIdx.x%64,wave=threadIdx.x/64,wm=wave/2,wn=wave%2;
  if(lane==0){stamps[4*wave]=b;stamps[4*wave+1]=e;stamps[4*wave+2]=wb;stamps[4*wave+3]=we;}
#pragma unroll
  for(int r=0;r<16;++r) {
    int row=wm*32+8*(r/4)+4*(lane/32)+r%4,col=wn*32+lane%32;
    out[row*64+col]=c[r];
  }
}
template<bool Hot> __global__ __launch_bounds__(256) void lds_grid(float* out,uint64_t* stamps,int steps,int records,uint32_t sa,uint32_t sb) {
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  for(int i=threadIdx.x;i<records*512;i+=256) smem[i]=local_word(i,sa,sb);
  __syncthreads();
  int lane=threadIdx.x%64,wave=threadIdx.x/64;
  uint32_t base=uint32_t(reinterpret_cast<uintptr_t>(smem));
  uint32_t alo=base+(wave/2)*512+lane*8,blo=base+1024+(wave%2)*512+lane*8;
  F16 c={};uint64_t b,e,wb,we;
  begin_clock(c,b,wb);compute<Hot>(alo,blo,steps,records,c);end_clock(c,e,we);
  save(c,out,stamps,b,e,wb,we);
}
// Packed input and LDS use the same per-K8 [A0,A1,B0,B1] block layout.
// Producer wave0/1 owns A0/A1; wave2/3 owns B0/B1.
__device__ __forceinline__ void produce(const uint32_t* src,uint32_t* dst) {
  int lane=threadIdx.x%64,wave=threadIdx.x/64;
  const uint32_t base=uint32_t(reinterpret_cast<uintptr_t>(dst));
  const uintptr_t p=reinterpret_cast<uintptr_t>(src+wave*128+lane);
  const uint32_t m=__builtin_amdgcn_readfirstlane(base+wave*512);
  copy_direct(uint32_t(p),uint32_t(p>>32),m);
}
__device__ __forceinline__ void finish_produce() {
  asm volatile("s_waitcnt vmcnt(0)" : : : "memory");
}
__global__ __launch_bounds__(256) void copy_check(const uint32_t* src,uint32_t* output) {
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  constexpr int words=kStageK/8*512;
  produce(src,smem);finish_produce();__syncthreads();
  produce(src+words,smem+words);finish_produce();__syncthreads();
  for(int i=threadIdx.x;i<2*words;i+=256) output[i]=smem[i];
}
template<int Buffers> __global__ __launch_bounds__(256) void gemm(const uint32_t* packed,float* out,uint64_t* stamps,int tiles) {
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  constexpr int words=kStageK/8*512;
  int lane=threadIdx.x%64,wave=threadIdx.x/64;
  F16 c={};uint64_t b,e,wb,we;
  begin_clock(c,b,wb);
  if(tiles) {
    produce(packed,smem);finish_produce();__syncthreads();
#pragma clang loop unroll(disable)
    for(int tile=0;tile<tiles;++tile) {
      if constexpr(Buffers==2) {
        if(tile+1<tiles) produce(packed+size_t(tile+1)*words,smem+((tile+1)%2)*words);
        // Future direct writes are VMEM; current LDS reads use LGKM.
      }
      const uint32_t base=uint32_t(reinterpret_cast<uintptr_t>(smem+(tile%Buffers)*words));
      compute<false>(base+(wave/2)*512+lane*8,base+1024+(wave%2)*512+lane*8,kStageK/8,kStageK/8,c);
      if constexpr(Buffers==2) { if(tile+1<tiles) finish_produce(); }
      // compute drained each wave's LDS reads; producer completion plus this
      // CTA barrier publishes next tile AND prevents overwriting current tile.
      __syncthreads();
      if constexpr(Buffers==1) {
        if(tile+1<tiles) {produce(packed+size_t(tile+1)*words,smem);finish_produce();__syncthreads();}
      }
    }
  }
  end_clock(c,e,we);save(c,out,stamps,b,e,wb,we);
}
template<class T>struct Device {
  T* p=nullptr;explicit Device(size_t n){HIP(hipMalloc(&p,std::max(size_t(1),n)*sizeof(T)));}
  ~Device(){if(p)(void)hipFree(p);}Device(const Device&)=delete;
};
struct Events {hipEvent_t b{},e{};Events(){HIP(hipEventCreate(&b));HIP(hipEventCreate(&e));}~Events(){(void)hipEventDestroy(b);(void)hipEventDestroy(e);}};
void verify_copy(uint32_t sa,uint32_t sb) {
  Data d(2*kStageK,sa,sb);const auto packed=d.pack(2*kStageK);
  Device<uint32_t> input(packed.size()),output(packed.size()+32);
  HIP(hipMemcpy(input.p,packed.data(),packed.size()*4,hipMemcpyHostToDevice));
  HIP(hipMemset(output.p,0xff,(packed.size()+32)*4));
  copy_check<<<1,256,packed.size()*4>>>(input.p,output.p+16);HIP(hipGetLastError());HIP(hipDeviceSynchronize());
  std::vector<uint32_t> y(packed.size()+32);HIP(hipMemcpy(y.data(),output.p,y.size()*4,hipMemcpyDeviceToHost));
  int bad=0;
  for(size_t i=0;i<packed.size();++i) if(y[i+16]!=packed[i]) {
    if(bad<4)std::fprintf(stderr,"COPY MISMATCH i=%zu expected=%08x actual=%08x\n",i,packed[i],y[i+16]);++bad;
  }
  for(int i=0;i<16;++i)bad+=y[i]!=0xffffffffu || y[y.size()-16+i]!=0xffffffffu;
  if(bad)throw std::runtime_error("global-to-LDS full-word copy/guard check failed");
  std::fprintf(stderr,"COPY CHECK direct-to-LDS words=%zu all exact, both LDS buffers and guards\n",packed.size());
}
int main(int argc,char** argv)try{
  std::string mode="gemm";
  const std::string producer="direct"; // Constant CSV provenance, not a selectable path.
  int steps=4096,records=0,rounds=9,warmup=200,verify=1,K=32768,buffers=2;
  uint32_t sa=1729,sb=2718;
  bool steps_set=false,records_set=false,buffers_set=false;
  for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){std::printf("--mode gemm|lds-hot|lds-sweep --buffers 1|2 --k N --steps N --records N --rounds N --warmup-ms N --verify 0|1 --seed-a N --seed-b N\nD=%d U=%d stageK=%d\n",kDepth,kUnroll,kStageK);return 0;}if(++i==argc)throw std::runtime_error("missing argument");std::string v=argv[i];
    if(key=="--mode")mode=v;else if(key=="--buffers"){buffers=std::stoi(v);buffers_set=true;}else if(key=="--k")K=std::stoi(v);else if(key=="--steps"){steps=std::stoi(v);steps_set=true;}else if(key=="--records"){records=std::stoi(v);records_set=true;}else if(key=="--rounds")rounds=std::stoi(v);else if(key=="--warmup-ms")warmup=std::stoi(v);else if(key=="--verify")verify=std::stoi(v);else if(key=="--seed-a")sa=uint32_t(std::stoul(v));else if(key=="--seed-b")sb=uint32_t(std::stoul(v));else throw std::runtime_error("unknown option "+key);
  }
  const bool real=mode=="gemm";
  if((!real&&mode!="lds-hot"&&mode!="lds-sweep")||(buffers!=1&&buffers!=2)||rounds<1||warmup<0||verify<0||verify>1||K<0||K>(1<<20)||(verify&&K>65536))throw std::runtime_error("invalid configuration");
  if(real&&(steps_set||records_set))throw std::runtime_error("GEMM uses --k and compiled stageK, not LDS --steps/--records");
  if(!real&&buffers_set)throw std::runtime_error("--buffers applies only to GEMM");
  const int tiles=(K+kStageK-1)/kStageK;
  if(real) { steps=tiles*(kStageK/8);records=kStageK/8; }
  else {
    if(steps<=0||steps%kUnroll||(verify&&steps>8192))throw std::runtime_error("invalid LDS steps");
    if(!records)records=mode=="lds-hot"?1:32;
    if(records<1||records>32||(mode=="lds-hot"&&records!=1))throw std::runtime_error("LDS records 1..32; hot=1");
  }
  const int logical_k=real?K:steps*8;
  const int shared_bytes=real?buffers*kStageK*256:records*2048;
  HIP(hipSetDevice(0));int wallrate=0;HIP(hipDeviceGetAttribute(&wallrate,hipDeviceAttributeWallClockRate,0));
  Data data(real?K:records*8,sa,sb);
  std::vector<float> ref;
  if(verify){
    ref=real?data.reference():data.reference(steps,records);
    std::vector<float> test(4128,std::bit_cast<float>(0xffffffffu));
    std::copy(ref.begin(),ref.end(),test.begin()+16);
    Check exact=check_output(test,ref,false);
    if(exact.bad||exact.guards)throw std::runtime_error("checker exact-result control failed");
    test[16]=ref[0]+1;
    if(check_output(test,ref,false).bad!=1)throw std::runtime_error("checker corruption control failed");
    test[0]=0;
    if(check_output(test,ref,false).guards!=1)throw std::runtime_error("checker guard control failed");
  }
  auto packed=real?data.pack(tiles*kStageK):std::vector<uint32_t>{};
  Device<uint32_t> input(packed.size());if(!packed.empty())HIP(hipMemcpy(input.p,packed.data(),packed.size()*4,hipMemcpyHostToDevice));
  if(real&&verify)verify_copy(sa,sb);
  Device<float> output(4128);Device<uint64_t> stamps(16);
  auto run=[&](){
    if(mode=="lds-hot")lds_grid<true><<<1,256,shared_bytes>>>(output.p+16,stamps.p,steps,records,sa,sb);
    else if(mode=="lds-sweep")lds_grid<false><<<1,256,shared_bytes>>>(output.p+16,stamps.p,steps,records,sa,sb);
    else if(buffers==1)gemm<1><<<1,256,shared_bytes>>>(input.p,output.p+16,stamps.p,tiles);
    else gemm<2><<<1,256,shared_bytes>>>(input.p,output.p+16,stamps.p,tiles);
    HIP(hipGetLastError());
  };
  std::fprintf(stderr,"grid=1 block=256 waves=2x2 mode=%s producer=%s buffers=%d D=%d U=%d stageK=%d K=%d executed_steps=%d records=%d LDS_bytes=%d packed_bytes=%zu seeds=%u/%u unique_input_B_per_step=2048 LDS_result_B_per_step=4096; native O3 HIP0 CPU+memory placement external no model/context; packing outside timing\n",mode.c_str(),real?producer.c_str():"none",real?buffers:0,kDepth,kUnroll,kStageK,logical_k,steps,records,shared_bytes,packed.size()*4,sa,sb);
  auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warmup);while(std::chrono::steady_clock::now()<until){run();HIP(hipDeviceSynchronize());}
  Events ev;std::printf("round,mode,producer,buffers,depth,unroll,stage_k,logical_k,steps,records,shared_bytes,packed_bytes,launch_us,cta_cycles,cycles_per_mfma,body_us,core_MHz,logical_GFLOPs,mismatches,guards,max_abs_error,valid");
  for(int w=0;w<4;++w)std::printf(",begin%d,end%d,wall_begin%d,wall_end%d",w,w,w,w);std::puts("");int invalid=0;
  for(int r=0;r<rounds;++r){HIP(hipMemset(output.p,0xff,4128*sizeof(float)));HIP(hipEventRecord(ev.b));run();HIP(hipEventRecord(ev.e));HIP(hipEventSynchronize(ev.e));float ms;HIP(hipEventElapsedTime(&ms,ev.b,ev.e));uint64_t st[16];HIP(hipMemcpy(st,stamps.p,sizeof(st),hipMemcpyDeviceToHost));uint64_t first=UINT64_MAX,last=0,wfirst=UINT64_MAX,wlast=0;for(int w=0;w<4;++w){first=std::min(first,st[w*4]);last=std::max(last,st[w*4+1]);wfirst=std::min(wfirst,st[w*4+2]);wlast=std::max(wlast,st[w*4+3]);}double cycles=double(last-first),us=double(wlast-wfirst)*1000/wallrate;
    Check c;if(verify){std::vector<float> y(4128);HIP(hipMemcpy(y.data(),output.p,y.size()*4,hipMemcpyDeviceToHost));c=check_output(y,ref);}invalid+=bool(c.bad||c.guards);
    std::printf("%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%.6f,%.0f,%.9f,%.9f,%.3f,%.6f,%d,%d,%.9g,%d",r,mode.c_str(),real?producer.c_str():"none",real?buffers:0,kDepth,kUnroll,kStageK,logical_k,steps,records,shared_bytes,packed.size()*4,ms*1000,cycles,steps?cycles/steps:0,us,cycles/us,8192.0*logical_k/(us*1000),c.bad,c.guards,c.maxerr,verify?int(!c.bad&&!c.guards):-1);
    for(int w=0;w<4;++w)std::printf(",%llu,%llu,%llu,%llu",(unsigned long long)st[4*w],(unsigned long long)st[4*w+1],(unsigned long long)st[4*w+2],(unsigned long long)st[4*w+3]);std::puts("");std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE invalid=%d verified=%d\n",invalid,verify);return invalid?2:0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
