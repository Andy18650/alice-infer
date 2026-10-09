// Single-GCD entry-timestamp probe. Sample before any CTA barrier; one record
// per wave (SIMT lanes share execution), not a counter/atomic shared by CTAs.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Stamp {
  uint64_t start_cycles, start_steady, end_cycles, end_steady;
  uint32_t cu, cooperative;
};
__global__ void k_stamp(Stamp* records, uint32_t ordinal, uint64_t hold_ticks) {
  const uint64_t start_cycles = clock64();
  const uint64_t start_steady = wall_clock64();
  const uint32_t cu = __smid();
  const uint32_t waves = blockDim.x / 64;
  const size_t index = (size_t(ordinal) * gridDim.x + blockIdx.x) * waves + threadIdx.x / 64;
  if ((threadIdx.x & 63) == 0) {
    records[index].start_cycles = start_cycles;
    records[index].start_steady = start_steady;
    records[index].cu = cu;
    records[index].cooperative = cooperative_groups::this_grid().is_valid();
  }
  // All waves hold, keeping their CTA resident. This is a scheduling control,
  // not a launch-latency subtraction. There is no entry synchronization.
  if (hold_ticks)
    while (uint64_t(wall_clock64()) - start_steady < hold_ticks) {}
  const uint64_t end_cycles = clock64();
  const uint64_t end_steady = wall_clock64();
  if ((threadIdx.x & 63) == 0) {
    records[index].end_cycles = end_cycles;
    records[index].end_steady = end_steady;
  }
}
void check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Case { int blocks, threads, launches; double hold_us; bool graph; bool cooperative = false; };
const char* mode_name(const Case& c) {
  if (c.cooperative) return c.graph ? "coop-graph" : "coop";
  return c.graph ? "graph" : "eager";
}
struct Context {
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
  int rate_khz = 0, cus = 0;
  bool cooperative = false;
  Context() {
    check(hipSetDevice(0), "select GCD 0");
    hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p,0), "properties");
    if (p.warpSize != 64) throw std::runtime_error("probe is compiled for gfx90a wave64");
    cus = p.multiProcessorCount; cooperative = p.cooperativeLaunch;
    check(hipDeviceGetAttribute(&rate_khz,hipDeviceAttributeWallClockRate,0), "steady clock rate");
    if (rate_khz <= 0) throw std::runtime_error("steady clock unavailable");
    check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking), "stream");
    check(hipEventCreateWithFlags(&start,hipEventDisableSystemFence), "start event");
    check(hipEventCreateWithFlags(&stop,hipEventDisableSystemFence), "stop event");
    std::fprintf(stderr,"GCD=0 arch=%s CUs=%d wave=%d cooperative=%d max_core_clock_khz=%d steady_clock_khz=%d\n",
                  p.gcnArchName,cus,p.warpSize,int(cooperative),p.clockRate,rate_khz);
  }
  ~Context() {
    if (start) (void)hipEventDestroy(start);
    if (stop) (void)hipEventDestroy(stop);
    if (stream) (void)hipStreamDestroy(stream);
  }
  double us(uint64_t ticks) const { return double(ticks)*1000/rate_khz; }
};
struct Work {
  Stamp* records = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  ~Work() {
    if (exec) (void)hipGraphExecDestroy(exec);
    if (graph) (void)hipGraphDestroy(graph);
    if (records) (void)hipFree(records);
  }
};
double median(std::vector<double> x) {
  std::sort(x.begin(),x.end());
  return x.size()%2 ? x[x.size()/2] : (x[x.size()/2-1]+x[x.size()/2])/2;
}
struct Timeline {
  double cta_spread = 0, wave_spread = 0, start_gap = 0;
  int max_cus = 0, early_finish = 0;
};
Timeline analyze(const std::vector<Stamp>& records,const Case& c,const Context& ctx) {
  Timeline result;
  const int waves = c.threads/64;
  const uint64_t hold = uint64_t(std::ceil(c.hold_us*ctx.rate_khz/1000));
  std::vector<uint64_t> first(c.launches);
  std::vector<double> spreads, wave_spreads;
  for (int kernel=0;kernel<c.launches;++kernel) {
    uint64_t minimum = UINT64_MAX, maximum = 0, wave_max = 0, earliest_cta_end = UINT64_MAX;
    std::set<uint32_t> cus;
    for (int cta=0;cta<c.blocks;++cta) {
      uint64_t begin=UINT64_MAX,end=0;
      for (int wave=0;wave<waves;++wave) {
        const auto& s=records[(size_t(kernel)*c.blocks+cta)*waves+wave];
        if (s.start_steady==UINT64_MAX || s.end_steady<s.start_steady || s.end_cycles<s.start_cycles ||
            s.end_steady-s.start_steady<hold)
          throw std::runtime_error("missing/invalid entry stamp or short hold");
        if (bool(s.cooperative)!=c.cooperative)
          throw std::runtime_error("device cooperative context does not match requested mode");
        if (s.cu!=records[(size_t(kernel)*c.blocks+cta)*waves].cu)
          throw std::runtime_error("waves of a CTA report different CUs");
        begin=std::min(begin,s.start_steady); end=std::max(end,s.end_steady);
        wave_max=std::max(wave_max,s.start_steady);cus.insert(s.cu);
      }
      minimum=std::min(minimum,begin);maximum=std::max(maximum,begin);
      earliest_cta_end=std::min(earliest_cta_end,end);
    }
    first[kernel]=minimum;
    spreads.push_back(ctx.us(maximum-minimum));
    wave_spreads.push_back(ctx.us(wave_max-minimum));
    result.max_cus=std::max(result.max_cus,int(cus.size()));
    if (earliest_cta_end<maximum) ++result.early_finish;
  }
  result.cta_spread=median(spreads);result.wave_spread=median(wave_spreads);
  if (c.launches>1) {
    std::vector<double> gaps;
    for (int k=1;k<c.launches;++k) {
      if (first[k]<first[k-1]) throw std::runtime_error("steady clock kernel order inconsistent");
      gaps.push_back(ctx.us(first[k]-first[k-1]));
    }
    result.start_gap=median(gaps);
  }
  return result;
}
hipError_t enqueue(Context& ctx,Work& w,const Case& c,uint64_t hold) {
  for (int k=0;k<c.launches;++k) {
    if (c.cooperative) {
      uint32_t ordinal = uint32_t(k);
      void* args[] = {&w.records, &ordinal, &hold};
      const auto e = hipLaunchCooperativeKernel(reinterpret_cast<const void*>(k_stamp),
                                                dim3(c.blocks), dim3(c.threads), args, 0, ctx.stream);
      if (e != hipSuccess) return e;
    } else k_stamp<<<c.blocks,c.threads,0,ctx.stream>>>(w.records,uint32_t(k),hold);
  }
  return hipGetLastError();
}

// Exercise the actual runtime rejection, not merely our occupancy guard.
// The kernel has no grid.sync(), so unexpected acceptance cannot deadlock;
// it is still a failed test rather than a normal-launch fallback.
bool oversized(Context& ctx,Work& work,const Case& c,uint64_t hold,int resident) {
  const char* stage = "launch";
  hipError_t status;
  if (c.graph) {
    check(hipStreamBeginCapture(ctx.stream,hipStreamCaptureModeThreadLocal),"oversize begin capture");
    status = enqueue(ctx,work,c,hold);
    const auto end = hipStreamEndCapture(ctx.stream,&work.graph);
    stage = "capture-launch";
    if (status == hipSuccess) { status = end; stage = "end-capture"; }
    if (status == hipSuccess) {
      status = hipGraphInstantiate(&work.exec,work.graph,nullptr,nullptr,0);
      stage = "instantiate";
    }
    if (status == hipSuccess) {
      status = hipGraphLaunch(work.exec,ctx.stream);
      stage = "replay";
    }
  } else status = enqueue(ctx,work,c,hold);
  if (status == hipSuccess) {
    check(hipStreamSynchronize(ctx.stream),"unexpected oversize completion");
    Stamp first{};
    check(hipMemcpy(&first,work.records,sizeof(first),hipMemcpyDeviceToHost),"oversize context read");
    std::printf("%s,%d,%d,%d,%.1f,0,%d,nan,nan,nan,nan,nan,0,0,UNEXPECTED_ACCEPTED_%s_device_coop%u\n",
                 mode_name(c),c.blocks,c.threads,c.launches,c.hold_us,resident,stage,first.cooperative);
    std::fflush(stdout);
    return false;  // reserved exit 2; recorded safely, never treated as a valid timing
  }
  if (status != hipErrorCooperativeLaunchTooLarge) check(status,"unexpected oversize error");
  (void)hipGetLastError();  // clear only the recognized, deliberately provoked error
  std::printf("%s,%d,%d,%d,%.1f,0,%d,nan,nan,nan,nan,nan,0,0,EXPECTED_FAIL_%s_%s\n",
               mode_name(c),c.blocks,c.threads,c.launches,c.hold_us,resident,stage,
               hipGetErrorName(status));
  std::fflush(stdout);
  return true;
}

void inspect_capture(const Work& work) {
  size_t count=0;check(hipGraphGetNodes(work.graph,nullptr,&count),"graph node count");
  std::vector<hipGraphNode_t> nodes(count);
  check(hipGraphGetNodes(work.graph,nodes.data(),&count),"graph nodes");
  int coop=0,ordinary=0;
  for (auto node:nodes) {
    hipGraphNodeType type{};check(hipGraphNodeGetType(node,&type),"graph node type");
    if (type!=hipGraphNodeTypeKernel) continue;
    hipKernelNodeAttrValue value{};
    check(hipGraphKernelNodeGetAttribute(node,hipKernelNodeAttributeCooperative,&value),"cooperative attribute");
    if (value.cooperative) ++coop; else ++ordinary;
  }
  std::fprintf(stderr,"captured kernel attributes: cooperative=%d ordinary=%d (device context checked separately)\n",coop,ordinary);
}

bool run(Context& ctx,const Case& c,int samples,const std::filesystem::path& out) {
  int active=0;
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,k_stamp,c.threads,0),"occupancy");
  const int resident=active*ctx.cus;
  if (c.cooperative && !ctx.cooperative) throw std::runtime_error("cooperative launch unsupported");
  const size_t per_kernel=size_t(c.blocks)*(c.threads/64);
  if (per_kernel>SIZE_MAX/size_t(c.launches)/sizeof(Stamp)) throw std::runtime_error("record size overflow");
  const size_t count=per_kernel*c.launches,bytes=count*sizeof(Stamp);
  const uint64_t hold=uint64_t(std::ceil(c.hold_us*ctx.rate_khz/1000));
  Work work;check(hipMalloc(&work.records,bytes),"record allocation");
  if (c.cooperative && c.blocks>resident) {
    return oversized(ctx,work,c,hold,resident);
  }
  if (c.graph) {
    check(hipStreamBeginCapture(ctx.stream,hipStreamCaptureModeThreadLocal),"begin capture");
    const auto e = enqueue(ctx,work,c,hold);
    const auto end = hipStreamEndCapture(ctx.stream,&work.graph);
    check(e,"captured stamp launch");
    check(end,"end capture");
    if (c.cooperative) inspect_capture(work);
    check(hipGraphInstantiate(&work.exec,work.graph,nullptr,nullptr,0),"instantiate");
  }
  std::vector<Stamp> records(count);
  std::vector<double> gpu,enqueue_times,cta,wave,gaps;
  int early_finish=0,cu_count=0;
  for (int sample=-3;sample<samples;++sample) {
    check(hipMemsetAsync(work.records,0xff,bytes,ctx.stream),"poison stamps");
    check(hipEventRecord(ctx.start,ctx.stream),"start event");
    const auto t0=std::chrono::steady_clock::now();
    if (c.graph) check(hipGraphLaunch(work.exec,ctx.stream),"graph launch");
    else check(enqueue(ctx,work,c,hold),"stamp launch");
    const auto t1=std::chrono::steady_clock::now();
    check(hipEventRecord(ctx.stop,ctx.stream),"stop event");
    check(hipStreamSynchronize(ctx.stream),"completion");
    if (sample<0) continue;
    float ms=0;check(hipEventElapsedTime(&ms,ctx.start,ctx.stop),"elapsed");
    check(hipMemcpy(records.data(),work.records,bytes,hipMemcpyDeviceToHost),"stamp download");
    const auto t=analyze(records,c,ctx);
    gpu.push_back(double(ms)*1000/c.launches);
    enqueue_times.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()/c.launches);
    cta.push_back(t.cta_spread);wave.push_back(t.wave_spread);gaps.push_back(t.start_gap);
    early_finish+=t.early_finish;cu_count=std::max(cu_count,t.max_cus);
  }
  const std::string tag=std::string(mode_name(c))+"-b"+std::to_string(c.blocks)+
      "-t"+std::to_string(c.threads)+"-n"+std::to_string(c.launches)+"-hold"+std::to_string(c.hold_us);
  std::ofstream raw(out/(tag+".csv"));
  raw<<"kernel,cta,wave,cu,device_cooperative,start_cycles,start_steady,end_cycles,end_steady,relative_start_us\n";
  const uint64_t origin=std::min_element(records.begin(),records.end(),[](const Stamp& a,const Stamp& b){return a.start_steady<b.start_steady;})->start_steady;
  for (int k=0;k<c.launches;++k)
    for (int b=0;b<c.blocks;++b)
      for (int w=0;w<c.threads/64;++w) {
        const auto& s=records[(size_t(k)*c.blocks+b)*(c.threads/64)+w];
        raw<<k<<','<<b<<','<<w<<','<<s.cu<<','<<s.cooperative<<','<<s.start_cycles<<','<<s.start_steady<<','
            <<s.end_cycles<<','<<s.end_steady<<','<<ctx.us(s.start_steady-origin)<<'\n';
      }
  if (!raw) throw std::runtime_error("raw trace write failed");
  std::printf("%s,%d,%d,%d,%.1f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,OK\n",
      mode_name(c),c.blocks,c.threads,c.launches,c.hold_us,samples,resident,
      median(gpu),median(enqueue_times),median(gaps),median(cta),median(wave),
      cu_count,early_finish);
  std::fflush(stdout);
  return true;
}
int integer(const char* text) {
  size_t used=0;const int value=std::stoi(text,&used);
  if (text[used] || value<1) throw std::runtime_error("invalid positive integer");
  return value;
}
}  // namespace

int main(int argc,char**argv) {
  try {
    Context context;
    const bool sweep=argc>=2 && std::string(argv[1])=="--sweep";
    const std::filesystem::path out=sweep?(argc>2?argv[2]:"temp/kernel-launch"):
                                                 (argc>7?argv[7]:"temp/kernel-launch");
    std::filesystem::create_directories(out);
    std::puts("mode,blocks,threads,launches,hold_us,samples,resident_limit,gpu_us_per_kernel,host_enqueue_us_per_kernel,start_gap_us,cta_entry_spread_us,wave_entry_spread_us,cus_seen,early_finish_kernels,correct");
    if (sweep) {
      if (argc>3) throw std::runtime_error("--sweep accepts only an optional output directory");
      int active=0;check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,k_stamp,256,0),"sweep occupancy");
      const int capacity=active*context.cus;
      bool supported=true;
      for (double hold:{0.0,100.0})
        for (int blocks:{1,8,32,context.cus,capacity,2*capacity})
          for (bool cooperative:{false,true})
            for (bool graph:{false,true})
              supported=run(context,{blocks,256,1,hold,graph,cooperative},7,out) && supported;
      for (int launches:{8,64,256})
        for (bool cooperative:{false,true})
          for (bool graph:{false,true})
            supported=run(context,{1,256,launches,0,graph,cooperative},7,out) && supported;
      if (!supported) return 2;
    } else {
      if (argc<6 || argc>8) throw std::runtime_error("usage: probe blocks threads launches hold_us eager|graph|coop|coop-graph [samples=7] [out_dir=temp/kernel-launch], or --sweep [out_dir]");
      const int blocks=integer(argv[1]),threads=integer(argv[2]),launches=integer(argv[3]);
      size_t used=0;const double hold=std::stod(argv[4],&used);
      const std::string mode=argv[5];
      if (threads%64 || threads>1024 || argv[4][used] || !std::isfinite(hold) || hold<0 || hold>1000000 ||
          (mode!="eager" && mode!="graph" && mode!="coop" && mode!="coop-graph"))
        throw std::runtime_error("invalid geometry/hold/mode");
      if (!run(context,{blocks,threads,launches,hold,mode=="graph" || mode=="coop-graph",
                         mode=="coop" || mode=="coop-graph"},argc>6?integer(argv[6]):7,out)) return 2;
    }
  } catch (const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
