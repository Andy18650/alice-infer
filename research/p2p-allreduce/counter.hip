// Persistent counter visibility probe. One counter per GCD, not a contended
// global counter. No payload, collective library, host barrier, or kernel
// boundary in the measured loop. Steady-clock deltas are device-local.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
enum class Poll { Serial, Parallel, Neighbor };
struct Result { uint64_t ticks, final; };

// The arrays are statically indexed/unrolled. Only parallel polling lanes
// access their assigned peer; serial polling intentionally visits peers in
// rank order, like the engine's barrier controller.
template <Poll Mode, int Order>
__device__ __forceinline__ void cycle(uint64_t* mine, uint64_t* const (&peer)[8],
                                     uint64_t* watched, int rank, int ws, uint64_t value) {
  if (threadIdx.x == 0)
    (void)__hip_atomic_fetch_add(reinterpret_cast<unsigned long long*>(mine), 1ULL,
                                 Order, __HIP_MEMORY_SCOPE_SYSTEM);
  if constexpr (Mode == Poll::Parallel) {
    __syncthreads();
    if (threadIdx.x < static_cast<unsigned>(ws) && int(threadIdx.x) != rank) {
      auto* flag = reinterpret_cast<unsigned long long*>(watched);
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value) {}
    }
    __syncthreads();
  } else if constexpr (Mode == Poll::Neighbor) {
    if (ws > 1) {
      auto* flag = reinterpret_cast<unsigned long long*>(watched);
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value) {}
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j >= ws || j == rank) continue;
      auto* flag = reinterpret_cast<unsigned long long*>(peer[j]);
      while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < value) {}
    }
  }
}

template <Poll Mode, int Order>
__global__ void k_counter(uint64_t* const* flags, Result* result, int rank, int ws,
                           uint64_t target, uint64_t warm) {
  if constexpr (Mode != Poll::Parallel) if (threadIdx.x != 0) return;
  uint64_t* mine = flags[rank];
  uint64_t* peer[8] = {};
  uint64_t* watched = nullptr;
  if constexpr (Mode == Poll::Serial) {
#pragma unroll
    for (int j = 0; j < 8; ++j) if (j < ws) peer[j] = flags[j];
  }
  if constexpr (Mode == Poll::Parallel)
    if (threadIdx.x < static_cast<unsigned>(ws)) watched = flags[threadIdx.x];
  if constexpr (Mode == Poll::Neighbor) watched = flags[(rank + ws - 1) % ws];
  for (uint64_t value = 1; value <= warm; ++value)
    cycle<Mode, Order>(mine, peer, watched, rank, ws, value);
  uint64_t start = 0;
  if (threadIdx.x == 0) {
    // A discarded RMW return can be queued. Drain warmup before timing,
    // especially the single-rank control with no remote dependency.
    while (__hip_atomic_load(reinterpret_cast<unsigned long long*>(mine),
                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < warm) {}
    start = wall_clock64();
  }
  for (uint64_t step = 1; step <= target; ++step)
    cycle<Mode, Order>(mine, peer, watched, rank, ws, warm + step);
  if (threadIdx.x == 0) {
    uint64_t final = 0;
    do {
      final = __hip_atomic_load(reinterpret_cast<unsigned long long*>(mine),
                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    } while (final < warm + target);
    const uint64_t end = wall_clock64();  // measure reaching N, not just issuing N
    result->ticks = end - start;
    result->final = final;
  }
}
using Launch = hipError_t (*)(hipStream_t, uint64_t* const*, Result*, int, int, uint64_t, uint64_t);
template <Poll Mode, int Order>
hipError_t launch(hipStream_t stream, uint64_t* const* flags, Result* result,
                  int rank, int ws, uint64_t target, uint64_t warm) {
  k_counter<Mode, Order><<<1, 64, 0, stream>>>(flags, result, rank, ws, target, warm);
  return hipGetLastError();
}
struct Variant { const char* poll; const char* order; Launch launch; };
const Variant variants[] = {
    {"serial", "relaxed", launch<Poll::Serial, __ATOMIC_RELAXED>},
    {"parallel", "relaxed", launch<Poll::Parallel, __ATOMIC_RELAXED>},
    {"neighbor", "relaxed", launch<Poll::Neighbor, __ATOMIC_RELAXED>},
    {"serial", "release", launch<Poll::Serial, __ATOMIC_RELEASE>},
    {"parallel", "release", launch<Poll::Parallel, __ATOMIC_RELEASE>},
    {"neighbor", "release", launch<Poll::Neighbor, __ATOMIC_RELEASE>},
};
void check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Rank {
  int device = 0, rate_khz = 0;
  hipStream_t stream = nullptr;
  uint64_t *counter = nullptr, **flags = nullptr;
  Result* result = nullptr;
  ~Rank() {
    (void)hipSetDevice(device);
    if (stream) (void)hipStreamDestroy(stream);
    if (counter) (void)hipFree(counter);
    if (flags) (void)hipFree(flags);
    if (result) (void)hipFree(result);
  }
};
uint64_t number(const char* text) {
  if (text[0] == '-') throw std::runtime_error("negative argument");
  size_t used = 0; const auto n = std::stoull(text, &used);
  if (text[used]) throw std::runtime_error("invalid argument");
  return n;
}
int integer(const char* text) {
  const auto n = number(text);
  if (n > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error("integer overflow");
  return int(n);
}
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  return x.size() % 2 ? x[x.size()/2] : (x[x.size()/2-1] + x[x.size()/2]) / 2;
}
void study(int ws, uint64_t target, int samples, int base, const Variant& v, uint64_t warm) {
  int devices = 0; check(hipGetDeviceCount(&devices), "device count");
  if (ws < 1 || ws > 8 || base > devices || ws > devices - base || !target || samples < 1 ||
      warm >= std::numeric_limits<uint64_t>::max() - target)
    throw std::runtime_error("invalid ranks/target/samples/device range/warmup");
  // PRE-alloc, serial owner-thread peer enable. Every rank completes this pass
  // before any counter, pointer table or result allocation is created.
  for (int r = 0; r < ws; ++r) {
    check(hipSetDevice(base + r), "peer device");
    for (int p = 0; p < ws; ++p) {
      if (p == r) continue;
      int can = 0; check(hipDeviceCanAccessPeer(&can, base+r, base+p), "peer query");
      if (!can) throw std::runtime_error("required peer access unavailable");
      const auto e = hipDeviceEnablePeerAccess(base+p, 0);
      if (e != hipErrorPeerAccessAlreadyEnabled) check(e, "peer enable");
    }
  }
  std::vector<std::unique_ptr<Rank>> ranks;
  std::vector<uint64_t*> flags;
  for (int r = 0; r < ws; ++r) {
    auto rank = std::make_unique<Rank>(); rank->device = base + r;
    check(hipSetDevice(rank->device), "allocate device");
    check(hipDeviceGetAttribute(&rank->rate_khz, hipDeviceAttributeWallClockRate, rank->device), "steady clock rate");
    if (rank->rate_khz <= 0) throw std::runtime_error("device steady clock unavailable");
    check(hipStreamCreateWithFlags(&rank->stream, hipStreamNonBlocking), "stream");
    check(hipExtMallocWithFlags(reinterpret_cast<void**>(&rank->counter), sizeof(uint64_t),
                                hipDeviceMallocFinegrained), "fine-grained counter");
    check(hipMalloc(&rank->flags, size_t(ws) * sizeof(uint64_t*)), "peer table");
    check(hipMalloc(&rank->result, sizeof(Result)), "result");
    flags.push_back(rank->counter);
    std::fprintf(stderr,"rank=%d device=%d steady_clock_khz=%d\n",r,rank->device,rank->rate_khz);
    ranks.push_back(std::move(rank));
  }
  for (auto& r : ranks) {
    check(hipSetDevice(r->device), "table device");
    check(hipMemcpy(r->flags, flags.data(), size_t(ws)*sizeof(uint64_t*), hipMemcpyHostToDevice), "peer upload");
  }
  std::vector<double> elapsed, wall;
  std::vector<std::vector<double>> per_rank(ws);
  for (int sample = 0; sample < samples; ++sample) {
    // Complete every reset before launching ANY rank. No flag reset races.
    for (auto& r : ranks) {
      check(hipSetDevice(r->device), "reset device");
      check(hipMemsetAsync(r->counter, 0, sizeof(uint64_t), r->stream), "counter reset");
    }
    for (auto& r : ranks) {
      check(hipSetDevice(r->device), "reset wait device");
      check(hipStreamSynchronize(r->stream), "reset completion");
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < ws; ++r) {
      check(hipSetDevice(ranks[r]->device), "launch device");
      check(v.launch(ranks[r]->stream, ranks[r]->flags, ranks[r]->result, r, ws, target, warm), "counter launch");
    }
    for (auto& r : ranks) {
      check(hipSetDevice(r->device), "wait device");
      check(hipStreamSynchronize(r->stream), "counter completion");
    }
    wall.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count());
    double maximum = 0;
    for (int r = 0; r < ws; ++r) {
      check(hipSetDevice(ranks[r]->device), "read device");
      Result got{}; uint64_t final = 0;
      check(hipMemcpy(&got, ranks[r]->result, sizeof(got), hipMemcpyDeviceToHost), "result download");
      check(hipMemcpy(&final, ranks[r]->counter, sizeof(final), hipMemcpyDeviceToHost), "counter download");
      if (final != warm + target || got.final != final || !got.ticks)
        throw std::runtime_error("incorrect final counter/timing at rank " + std::to_string(r));
      const double us = double(got.ticks) * 1000 / ranks[r]->rate_khz;
      per_rank[r].push_back(us / target);
      maximum = std::max(maximum, us);
    }
    elapsed.push_back(maximum);
  }
  auto sorted = elapsed; std::sort(sorted.begin(),sorted.end());
  std::puts("poll,increment,ws,device_base,target,warmup,samples,device_median_us,device_min_us,device_p90_us,us_per_increment,wall_median_us,correct");
  std::printf("%s,%s,%d,%d,%llu,%llu,%d,%.4f,%.4f,%.4f,%.6f,%.4f,OK\n",
      v.poll,v.order,ws,base,(unsigned long long)target,(unsigned long long)warm,samples,
      median(elapsed),sorted.front(),sorted[size_t(std::ceil(.9*sorted.size()))-1],
      median(elapsed)/target,median(wall));
  for (int r=0;r<ws;++r)
    std::fprintf(stderr,"rank=%d median_us_per_increment=%.6f final=%llu\n",r,
                  median(per_rank[r]),(unsigned long long)(warm+target));
}
}  // namespace

int main(int argc,char**argv) {
  if(argc<2 || argc>8) {
    std::fprintf(stderr,"usage: %s ws [target=10000] [samples=7] [device_base=0] [serial|parallel|neighbor] [relaxed|release] [warmup=64]\n",argv[0]);
    return 1;
  }
  try {
    const std::string poll=argc>5?argv[5]:"parallel", order=argc>6?argv[6]:"relaxed";
    const Variant* variant=nullptr;
    for(const auto& v:variants) if(poll==v.poll && order==v.order) variant=&v;
    if(!variant) throw std::runtime_error("invalid polling/increment mode");
    study(integer(argv[1]),argc>2?number(argv[2]):10000,argc>3?integer(argv[3]):7,
          argc>4?integer(argv[4]):0,*variant,argc>7?number(argv[7]):64);
  } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
