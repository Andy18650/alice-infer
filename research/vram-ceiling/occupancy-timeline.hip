// Timestamped private twin of read_stream<2,16,false>. No CTA barriers,
// holds, atomics, or cooperative launch. Instrumentation is not the original.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
struct Stamp { uint64_t start,end; uint32_t cu,pad; };

__global__ void initialize(uint32_t* p,size_t n) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
    p[i]=uint32_t(i)*1664525u+1013904223u;
}

__global__ void read_stamped(const uint2* __restrict__ p,size_t nvec,
                             uint32_t* __restrict__ sink,Stamp* __restrict__ records) {
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const uint64_t start=wall_clock64();
  if((threadIdx.x&63)==0) {
    records[wave].start=start;
    records[wave].cu=__smid();
  }
  // Same input traversal, accumulation, loop unroll, tails and checksum
  // as the exact original kernel. Marker stores are the only added work.
  uint32_t a[16][2]={};
  const size_t step=size_t(gridDim.x)*blockDim.x;
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  for(;i+15*step<nvec;i+=16*step) {
#pragma unroll
    for(int u=0;u<16;++u) {
      auto v=p[i+u*step]; a[u][0]+=v.x; a[u][1]+=v.y;
    }
  }
  for(;i<nvec;i+=step) { auto v=p[i]; a[0][0]+=v.x; a[0][1]+=v.y; }
  uint32_t sum=0;
#pragma unroll
  for(int u=0;u<16;++u) sum+=a[u][0]+a[u][1];
  for(int delta=32;delta;delta/=2) sum+=__shfl_down(sum,delta,64);
  if((threadIdx.x&63)==0) sink[wave]=sum;
  const uint64_t end=wall_clock64();
  if((threadIdx.x&63)==0) records[wave].end=end;
}

using Interval=std::pair<uint64_t,uint64_t>;
int peak(const std::vector<Interval>& intervals) {
  std::vector<std::pair<uint64_t,int>> events;
  for(auto [s,e]:intervals) { events.push_back({s,1}); events.push_back({e,-1}); }
  std::sort(events.begin(),events.end());  // ends before starts at equal ticks
  int active=0,maximum=0;
  for(auto [tick,delta]:events) { (void)tick; active+=delta; maximum=std::max(maximum,active); }
  return maximum;
}

int main(int argc,char** argv) try {
  const std::string out=argc>1?argv[1]:"temp/vram-ceiling/occupancy/timeline";
  std::filesystem::create_directories(out);
  HIP(hipSetDevice(0));
  hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  int rate; HIP(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0));
  if(prop.warpSize!=64 || rate<=0) throw std::runtime_error("requires wave64 and a steady clock");
  hipFuncAttributes attributes{}; HIP(hipFuncGetAttributes(&attributes,reinterpret_cast<const void*>(read_stamped)));
  int active; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,read_stamped,256,0));
  std::ofstream meta(out+"/device.txt");
  meta<<"device=0 PCI="<<pci<<" CUs="<<prop.multiProcessorCount<<" wave="<<prop.warpSize
      <<" wall_clock_khz="<<rate<<" timestamped_regs="<<attributes.numRegs
      <<" shared="<<attributes.sharedSizeBytes<<" local="<<attributes.localSizeBytes
      <<" active_CTAs_per_CU="<<active<<'\n'; meta.close();
  std::printf("timestamped twin: regs=%d shared=%zu local=%zu CTAs/CU=%d capacity=%d\n",
              attributes.numRegs,attributes.sharedSizeBytes,attributes.localSizeBytes,active,active*prop.multiProcessorCount);
  hipStream_t stream; HIP(hipStreamCreate(&stream));
  hipEvent_t start,end; HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
  const size_t max_bytes=size_t(4)<<30;
  uint32_t *p,*sink; Stamp* stamps;
  HIP(hipMalloc(&p,max_bytes)); HIP(hipMalloc(&sink,1664*4*4)); HIP(hipMalloc(&stamps,1664*4*sizeof(Stamp)));
  initialize<<<prop.multiProcessorCount*8,256,0,stream>>>(p,max_bytes/4);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  std::ofstream summary(out+"/summary.csv");
  summary<<"bytes,blocks,sample,event_us,cus,peak_CTAs,peak_waves,max_CTAs_one_CU,max_waves_one_CU,CTA_entry_spread_us,all_CTA_common_interval_us,all_wave_common_interval_us\n";
  for(size_t bytes:{size_t(1)<<30,size_t(4)<<30}) for(int blocks:{416,832,1664}) {
    for(int i=0;i<20;++i) read_stamped<<<blocks,256,0,stream>>>(reinterpret_cast<const uint2*>(p),bytes/8,sink,stamps);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
    for(int sample=0;sample<7;++sample) {
      HIP(hipMemsetAsync(stamps,0,blocks*4*sizeof(Stamp),stream));
      HIP(hipEventRecord(start,stream));
      read_stamped<<<blocks,256,0,stream>>>(reinterpret_cast<const uint2*>(p),bytes/8,sink,stamps);
      HIP(hipGetLastError()); HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
      float ms; HIP(hipEventElapsedTime(&ms,start,end));
      std::vector<Stamp> records(blocks*4);
      std::vector<uint32_t> sums(blocks*4);
      HIP(hipMemcpyAsync(records.data(),stamps,records.size()*sizeof(Stamp),hipMemcpyDeviceToHost,stream));
      HIP(hipMemcpyAsync(sums.data(),sink,sums.size()*4,hipMemcpyDeviceToHost,stream));
      HIP(hipStreamSynchronize(stream));
      uint32_t sum=0; for(auto v:sums) sum+=v;
      const uint64_t n=bytes/4;
      if(sum!=uint32_t(n*(n-1)/2)*1664525u+uint32_t(n)*1013904223u) throw std::runtime_error("checksum mismatch");
      std::vector<Interval> ctas,waves;
      std::map<uint32_t,std::vector<Interval>> by_cu,waves_by_cu;
      uint64_t first=UINT64_MAX,last=0,first_end=UINT64_MAX;
      uint64_t last_wave_start=0,first_wave_end=UINT64_MAX;
      std::ofstream raw(out+"/trace-"+std::to_string(bytes>>30)+"GiB-"+std::to_string(blocks)+"-"+std::to_string(sample)+".csv");
      raw<<"CTA,wave,CU,start_tick,end_tick\n";
      for(int block=0;block<blocks;++block) {
        uint64_t cstart=UINT64_MAX,cend=0;
        const auto cu=records[block*4].cu;
        for(int w=0;w<4;++w) {
          const auto& r=records[block*4+w];
          if(r.end<=r.start || r.cu!=cu) throw std::runtime_error("invalid markers/CU placement");
          cstart=std::min(cstart,r.start); cend=std::max(cend,r.end);
          waves.push_back({r.start,r.end}); waves_by_cu[cu].push_back({r.start,r.end});
          last_wave_start=std::max(last_wave_start,r.start); first_wave_end=std::min(first_wave_end,r.end);
          raw<<block<<','<<w<<','<<cu<<','<<r.start<<','<<r.end<<'\n';
        }
        ctas.push_back({cstart,cend}); by_cu[cu].push_back({cstart,cend});
        first=std::min(first,cstart); last=std::max(last,cstart); first_end=std::min(first_end,cend);
      }
      int maximum=0,max_waves=0;
      for(const auto& [cu,v]:by_cu) { (void)cu; maximum=std::max(maximum,peak(v)); }
      for(const auto& [cu,v]:waves_by_cu) { (void)cu; max_waves=std::max(max_waves,peak(v)); }
      const double common=first_end>last?double(first_end-last)*1000/rate:0;
      const double wave_common=first_wave_end>last_wave_start?double(first_wave_end-last_wave_start)*1000/rate:0;
      summary<<bytes<<','<<blocks<<','<<sample<<','<<ms*1000<<','<<by_cu.size()<<','<<peak(ctas)<<','<<peak(waves)
             <<','<<maximum<<','<<max_waves<<','<<double(last-first)*1000/rate<<','<<common<<','<<wave_common<<'\n';
      // A partially draining CTA can coexist with newly admitted waves from
      // another CTA. Whole-CTA occupancy arithmetic is not a bound on the
      // count of CTA intervals with just one wave still alive.
      if(max_waves>prop.maxThreadsPerMultiProcessor/64) throw std::runtime_error("wave intervals exceed device limit");
    }
    summary.flush(); std::printf("checked %zu GiB / %d CTAs\n",bytes>>30,blocks);
  }
  HIP(hipFree(stamps)); HIP(hipFree(sink)); HIP(hipFree(p));
  HIP(hipEventDestroy(start)); HIP(hipEventDestroy(end)); HIP(hipStreamDestroy(stream));
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
