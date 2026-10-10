// Local streaming-read ceiling; standalone HIP/C++20, no engine objects.
// All input words contribute exactly once to a checked modulo-2^32 sum.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP(x) do { auto e = (x); if (e != hipSuccess) \
  throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e)); } while (0)

template<int W> struct Vector;
template<> struct Vector<1> { using Type = uint32_t; };
template<> struct Vector<2> { using Type = uint2; };
template<> struct Vector<4> { using Type = uint4; };

__global__ void initialize(uint32_t* p, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint32_t(i) * 1664525u + 1013904223u;  // dispersed, deterministic words
}

// W: words/load, U: independent accumulator chains. Two traversals:
// global: whole-grid interleaved; slab: contiguous partition per CTA.
// Accumulate vector components separately to preserve wide loads in codegen.
template<int W, int U, bool Slab>
__global__ void read_stream(const typename Vector<W>::Type* __restrict__ p,
                            size_t nvec, uint32_t* __restrict__ sink) {
  uint32_t a[U][W] = {};
  const size_t begin = Slab ? (nvec * blockIdx.x / gridDim.x) : 0;
  const size_t end = Slab ? (nvec * (blockIdx.x + 1) / gridDim.x) : nvec;
  const size_t step = Slab ? blockDim.x : size_t(gridDim.x) * blockDim.x;
  size_t i = begin + (Slab ? threadIdx.x : size_t(blockIdx.x) * blockDim.x + threadIdx.x);
  for (; i + (U - 1) * step < end; i += U * step) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      auto v = p[i + u * step];
      if constexpr (W == 1) a[u][0] += v;
      else {
        a[u][0] += v.x; a[u][1] += v.y;
        if constexpr (W == 4) { a[u][2] += v.z; a[u][3] += v.w; }
      }
    }
  }
  for (; i < end; i += step) {
    auto v = p[i];
    if constexpr (W == 1) a[0][0] += v;
    else {
      a[0][0] += v.x; a[0][1] += v.y;
      if constexpr (W == 4) { a[0][2] += v.z; a[0][3] += v.w; }
    }
  }
  uint32_t sum = 0;
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int w = 0; w < W; ++w) sum += a[u][w];
  // One unique output per native wave; negligible store volume, no races.
  for (int delta = 32; delta; delta /= 2) sum += __shfl_down(sum, delta, 64);
  if ((threadIdx.x & 63) == 0)
    sink[(size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 64] = sum;
}

struct Case { int width, unroll, block, grid; bool slab; double score = 0; };

void launch(const Case& c, const uint32_t* p, size_t bytes, uint32_t* sink,
            hipStream_t stream) {
#define RUN(W,U) if (c.width == W && c.unroll == U) { \
  using T = typename Vector<W>::Type; \
  if (c.slab) read_stream<W,U,true><<<c.grid,c.block,0,stream>>>(reinterpret_cast<const T*>(p),bytes/(4*W),sink); \
  else read_stream<W,U,false><<<c.grid,c.block,0,stream>>>(reinterpret_cast<const T*>(p),bytes/(4*W),sink); \
  return; }
  RUN(1,1) RUN(1,4) RUN(1,8) RUN(1,16)
  RUN(2,1) RUN(2,4) RUN(2,8) RUN(2,16)
  RUN(4,1) RUN(4,4) RUN(4,8) RUN(4,16)
#undef RUN
  throw std::runtime_error("unsupported case");
}

void validate(const Case& c, size_t bytes, uint32_t* sink, hipStream_t stream) {
  const size_t n = bytes / 4, nsink = size_t(c.grid) * c.block / 64;
  std::vector<uint32_t> h(nsink);
  HIP(hipMemcpyAsync(h.data(), sink, nsink * 4, hipMemcpyDeviceToHost, stream));
  HIP(hipStreamSynchronize(stream));
  uint32_t sum = 0;
  for (auto v : h) sum += v;
  const uint32_t expected = uint32_t(uint64_t(n) * (n - 1) / 2) * 1664525u
                          + uint32_t(n) * 1013904223u;
  if (sum != expected) throw std::runtime_error("checksum mismatch");
}

struct Timer {
  hipStream_t stream{};
  hipEvent_t start{}, end{};
  Timer() { HIP(hipStreamCreate(&stream)); HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end)); }
  ~Timer() { (void)hipEventDestroy(start); (void)hipEventDestroy(end); (void)hipStreamDestroy(stream); }
  double measure(const Case& c, const uint32_t* p, size_t bytes, uint32_t* sink, int iters) {
    HIP(hipEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) launch(c, p, bytes, sink, stream);
    HIP(hipGetLastError());
    HIP(hipEventRecord(end, stream)); HIP(hipEventSynchronize(end));
    float ms; HIP(hipEventElapsedTime(&ms, start, end));
    return ms / iters;
  }
};

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }
void fields(std::ostream& o, const Case& c) {
  o << (c.slab ? "slab" : "global") << ',' << c.width * 4 << ',' << c.unroll << ',' << c.block << ',' << c.grid;
}

int main(int argc, char** argv) try {
  const std::string out = argc > 1 ? argv[1] : "temp/vram-ceiling/results";
  const int dev = argc > 2 ? std::stoi(argv[2]) : 0;
  std::filesystem::create_directories(out);
  HIP(hipSetDevice(dev));
  hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop, dev));
  int runtime, driver; HIP(hipRuntimeGetVersion(&runtime)); HIP(hipDriverGetVersion(&driver));
  char pci[64]{}; HIP(hipDeviceGetPCIBusId(pci, sizeof(pci), dev));
  std::ofstream meta(out + "/device.txt");
  meta << "device=" << dev << " PCI=" << pci << " name=" << prop.name << " arch=" << prop.gcnArchName
       << " CUs=" << prop.multiProcessorCount << " wave=" << prop.warpSize
       << " memory=" << prop.totalGlobalMem << " runtime=" << runtime << " driver=" << driver << '\n';
  meta.close();
  if (prop.warpSize != 64) throw std::runtime_error("probe requires wave64");
  Timer timer;
  const size_t max_bytes = size_t(4) << 30;
  uint32_t *p{}, *sink{};
  HIP(hipMalloc(&p, max_bytes));
  HIP(hipMalloc(&sink, size_t(prop.multiProcessorCount) * 32 * 512 / 64 * 4));
  initialize<<<prop.multiProcessorCount * 8,256,0,timer.stream>>>(p, max_bytes/4);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(timer.stream));
  std::vector<Case> cases;
  for (bool slab : {false,true}) for (int w : {1,2,4}) for (int u : {1,4,8,16})
    for (int b : {64,128,256,512}) for (int mult : {1,2,4,8,16,32})
      cases.push_back({w,u,b,prop.multiProcessorCount*mult,slab});
  std::mt19937 random(0x90a);
  std::shuffle(cases.begin(), cases.end(), random);
  // Exercise short loops, non-power-of-two lengths, and partition tails.
  for (size_t small : {size_t(64), size_t(4096 + 64), size_t((1 << 20) + 64)})
    for (const auto& c : cases) {
      launch(c,p,small,sink,timer.stream);
      HIP(hipGetLastError()); validate(c,small,sink,timer.stream);
    }
  std::fprintf(stderr,"all 576 variants passed three short/tail checksum controls\n");
  const size_t bytes = size_t(1) << 30;
  std::ofstream raw(out + "/sweep-raw.csv"), summary(out + "/sweep.csv");
  raw << "order,pattern,load_bytes,unroll,block,grid,bytes,sample,iters,ms,GB_s\n";
  summary << "pattern,load_bytes,unroll,block,grid,bytes,median_GB_s,min_GB_s,max_GB_s\n";
  // Preheat with real traffic, without changing clocks/power settings.
  const Case baseline{1,1,256,prop.multiProcessorCount*8,false};
  for (int i = 0; i < 250; ++i) launch(baseline,p,bytes,sink,timer.stream);
  HIP(hipStreamSynchronize(timer.stream));
  int order = 0;
  for (auto& c : cases) {
    for (int i = 0; i < 3; ++i) launch(c,p,bytes,sink,timer.stream);
    HIP(hipGetLastError()); validate(c,bytes,sink,timer.stream);
    std::vector<double> bw;
    for (int sample = 0; sample < 3; ++sample) {
      const double ms = timer.measure(c,p,bytes,sink,12), rate = bytes / (ms*1e6);
      bw.push_back(rate);
      raw << order << ','; fields(raw,c); raw << ',' << bytes << ',' << sample << ",12," << ms << ',' << rate << '\n';
    }
    c.score = median(bw);
    fields(summary,c); summary << ',' << bytes << ',' << c.score << ','
        << *std::min_element(bw.begin(),bw.end()) << ',' << *std::max_element(bw.begin(),bw.end()) << '\n';
    if (++order % 100 == 0) std::fprintf(stderr,"sweep %d/%zu\n",order,cases.size());
  }
  std::sort(cases.begin(),cases.end(),[](auto a,auto b){ return a.score > b.score; });
  std::fprintf(stderr,"screen best %.1f GB/s; confirming top 12 + baseline\n",cases.front().score);
  std::vector<Case> finalists(cases.begin(),cases.begin()+12);
  finalists.push_back(baseline);
  std::ofstream confirm(out + "/confirm-raw.csv");
  confirm << "pattern,load_bytes,unroll,block,grid,bytes,round,iters,ms,GB_s\n";
  // Larger-than-cache controls, repeated randomized paired rounds.
  for (size_t size : {size_t(1)<<30,size_t(2)<<30,size_t(4)<<30}) {
    for (int round = 0; round < 9; ++round) {
      std::shuffle(finalists.begin(),finalists.end(),random);
      for (const auto& c : finalists) {
        for (int i=0;i<3;++i) launch(c,p,size,sink,timer.stream);
        const double ms=timer.measure(c,p,size,sink,12);
        validate(c,size,sink,timer.stream);
        fields(confirm,c); confirm << ',' << size << ',' << round << ",12," << ms << ',' << size/(ms*1e6) << '\n';
      }
    }
    std::fprintf(stderr,"confirmed %zu GiB\n",size>>30);
  }
  HIP(hipFree(sink)); HIP(hipFree(p));
  return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
