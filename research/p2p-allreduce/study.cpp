// Out-of-place P2P research. Single host coordinator, one stream/graph per GCD.
// Immutable slots, live epochs, GPU-written changing inputs, all-rank completion.
// Latencies include the identical small producer in every backend; a fill-only
// control is provided, not silently subtracted as if costs were additive.
#include "kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using p2p_ar::Args;

void check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
struct Skip : std::runtime_error { using std::runtime_error::runtime_error; };

// Select one explicit launch function at startup; enqueue() has no backend
// switch, size-dependent dispatch, or second-level generic launcher.
struct Operation {
  const char* name;
  p2p_ar::Launch launch;
  int fixed_blocks, fixed_threads;
  bool has_result;
};
const Operation operations[] = {
    {"fill", p2p_ar::producer_only, 0, 0, false},
    {"fused", p2p_ar::fused, 1, 1024, true},
    {"pipe", p2p_ar::pipe, 0, 0, true},
    {"chunk", p2p_ar::chunk, 0, 0, true},
    {"hoisted", p2p_ar::hoisted, 0, 0, true},
    {"phase", p2p_ar::phase, 0, 0, true},
    {"peer-major", p2p_ar::peer_major, 0, 0, true},
    {"prefetch2", p2p_ar::prefetch2, 0, 0, true},
    {"prefetch4", p2p_ar::prefetch4, 0, 0, true},
    {"hoist-prefetch2", p2p_ar::hoist_prefetch2, 0, 0, true},
    {"peer-prefetch2", p2p_ar::peer_prefetch2, 0, 0, true},
    {"peer-prefetch4", p2p_ar::peer_prefetch4, 0, 0, true},
    {"peer-reg1", p2p_ar::peer_reg1, 0, 0, true},
    {"peer-reg4", p2p_ar::peer_reg4, 0, 0, true},
    {"peer-reg8", p2p_ar::peer_reg8, 0, 0, true},
    {"peer-reg32", p2p_ar::peer_reg32, 0, 0, true},
};
const Operation& operation(const std::string& name) {
  for (const auto& op : operations) if (name == op.name) return op;
  throw std::runtime_error("unknown operation: " + name);
}
struct Config {
  const Operation* op = nullptr;
  std::string mode = "graph";
  int ws = 0, blocks = 0, threads = 256, slots = 64, samples = 5, base = 0;
  size_t bytes = 0;
};
long long number(const char* text) {
  size_t used = 0;
  const auto n = std::stoll(text, &used);
  if (text[used] != '\0') throw std::runtime_error("invalid number: " + std::string(text));
  return n;
}
int integer(const char* text) {
  const auto n = number(text);
  if (n < std::numeric_limits<int>::min() || n > std::numeric_limits<int>::max())
    throw std::runtime_error("integer out of range");
  return int(n);
}
struct Graph {
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  ~Graph() {
    if (exec) (void)hipGraphExecDestroy(exec);
    if (graph) (void)hipGraphDestroy(graph);
  }
};
struct Rank {
  int device = 0;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
  float *input = nullptr, *out = nullptr;
  float** peers = nullptr;
  uint64_t *flag = nullptr, *epoch_base = nullptr;
  uint64_t** flags = nullptr;
  uint64_t host_base = 0;
  std::unique_ptr<Graph> validation, timed;
  ~Rank() {
    (void)hipSetDevice(device);
    validation.reset(); timed.reset();
    if (start) (void)hipEventDestroy(start);
    if (stop) (void)hipEventDestroy(stop);
    if (stream) (void)hipStreamDestroy(stream);
    if (input) (void)hipFree(input);
    if (out) (void)hipFree(out);
    if (peers) (void)hipFree(peers);
    if (flag) (void)hipFree(flag);
    if (flags) (void)hipFree(flags);
    if (epoch_base) (void)hipFree(epoch_base);
  }
};
using Ranks = std::vector<std::unique_ptr<Rank>>;

Args args_for(Rank& r, const Config& c, int rank, int slot, bool validation) {
  Args a;
  a.input = r.input; a.out = r.out;
  a.peers = r.peers; a.flag = r.flag; a.flags = r.flags;
  a.epoch_base = r.epoch_base;
  a.offset = size_t(slot) * (c.bytes / 4);
  if (validation) a.out += a.offset;
  a.ordinal = uint64_t(slot + 1);
  a.n = uint32_t(c.bytes / 4); a.rank = rank; a.ws = c.ws;
  return a;
}
void enqueue(Rank& r, const Config& c, int rank, int slots, bool validation) {
  for (int slot = 0; slot < slots; ++slot) {
    const auto a = args_for(r, c, rank, slot, validation);
    p2p_ar::fill(r.stream, a);  // prior kernel writes dirty L2; release is load-bearing
    const auto e = c.op->launch(r.stream, a, c.blocks, c.threads);
    check(e, c.op->name);
  }
}
std::unique_ptr<Graph> capture(Rank& r, const Config& c, int rank, int slots, bool validation) {
  auto g = std::make_unique<Graph>();
  check(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal), "begin capture");
  try { enqueue(r, c, rank, slots, validation); }
  catch (...) {
    (void)hipStreamEndCapture(r.stream, &g->graph);
    throw;
  }
  check(hipStreamEndCapture(r.stream, &g->graph), "end capture");
  check(hipGraphInstantiate(&g->exec, g->graph, nullptr, nullptr, 0), "instantiate");
  return g;
}
struct Times { double gpu = 0, wall = 0; };
Times run(Ranks& ranks, const Config& c, uint64_t base, int slots, bool validation) {
  for (auto& ptr : ranks) {
    auto& r = *ptr;
    check(hipSetDevice(r.device), "select device");
    r.host_base = base;
    check(hipMemcpyAsync(r.epoch_base, &r.host_base, sizeof(base), hipMemcpyHostToDevice,
                         r.stream), "epoch upload");
    if (validation)
      check(hipMemsetAsync(r.out, 0xff, c.bytes * size_t(slots), r.stream), "poison outputs");
    check(hipEventRecord(r.start, r.stream), "start event");
  }
  const auto t0 = std::chrono::steady_clock::now();
  // Submit every rank before waiting. No CPU barrier inside a collective.
  for (int rank = 0; rank < c.ws; ++rank) {
    auto& r = *ranks[rank];
    check(hipSetDevice(r.device), "select launch device");
    if (c.mode == "graph")
      check(hipGraphLaunch((validation ? r.validation : r.timed)->exec, r.stream), "graph launch");
    else enqueue(r, c, rank, slots, validation);
    check(hipEventRecord(r.stop, r.stream), "stop event");
  }
  for (auto& ptr : ranks) {
    check(hipSetDevice(ptr->device), "select wait device");
    check(hipStreamSynchronize(ptr->stream), "batch completion");
  }
  Times t;
  t.wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / slots;
  for (auto& ptr : ranks) {
    float ms = 0;
    check(hipSetDevice(ptr->device), "select event device");
    check(hipEventElapsedTime(&ms, ptr->start, ptr->stop), "elapsed time");
    t.gpu = std::max(t.gpu, double(ms) * 1000 / slots);
  }
  return t;
}
float input_value(int rank, uint64_t value, size_t i) {
  return float((rank + 1) * 128 + (value % 1024) * 4 + (i % 31)) * 0.03125f;
}
void verify(Ranks& ranks, const Config& c, uint64_t base, int slots, bool all_outputs) {
  const size_t n = c.bytes / 4;
  const size_t count = all_outputs ? n * size_t(slots) : n;
  std::vector<float> input(count), out(count);
  for (int rank = 0; rank < c.ws; ++rank) {
    auto& r = *ranks[rank];
    check(hipSetDevice(r.device), "select verification device");
    const size_t input_offset = all_outputs ? 0 : n * size_t(slots - 1);
    check(hipMemcpy(input.data(), r.input + input_offset, count * 4,
                    hipMemcpyDeviceToHost), "input download");
    const bool has_result = c.op->has_result;
    if (has_result)
      check(hipMemcpy(out.data(), r.out, count * 4, hipMemcpyDeviceToHost), "output download");
    for (int checked_slot = 0; checked_slot < (all_outputs ? slots : 1); ++checked_slot) {
      const int slot = all_outputs ? checked_slot : slots - 1;
      const uint64_t value = base + uint64_t(slot + 1);
      const size_t offset = size_t(checked_slot) * n;
      // All operands are small integer multiples of 1/32: this closed-form
      // sum is exactly the same as rank-ordered fp32 addition, not a tolerance.
      const uint64_t common = (value % 1024) * 4;
      const uint64_t sum_base = uint64_t(128) * c.ws * (c.ws + 1) / 2 + c.ws * common;
      for (size_t i = 0; i < n; ++i) {
        const size_t j = offset + i;
        if (input[j] != input_value(rank, value, i))
          throw std::runtime_error("input overwritten/stale at rank " + std::to_string(rank) +
                                   " slot " + std::to_string(slot) + " element " + std::to_string(i));
        const float want = float(sum_base + c.ws * (i % 31)) * 0.03125f;
        if (has_result && out[j] != want)
          throw std::runtime_error("incorrect sum at rank " + std::to_string(rank) +
                                   " slot " + std::to_string(slot) + " element " + std::to_string(i));
      }
    }
    if (c.op->launch != p2p_ar::producer_only) {
      uint64_t value = 0;
      check(hipMemcpy(&value, r.flag, sizeof(value), hipMemcpyDeviceToHost), "flag download");
      if (value != base + slots) throw std::runtime_error("incorrect final epoch");
    }
  }
}
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  return x.size() % 2 ? x[x.size() / 2] : (x[x.size() / 2 - 1] + x[x.size() / 2]) / 2;
}

void study(Config c) {
  int devices = 0;
  check(hipGetDeviceCount(&devices), "device count");
  if (c.ws < 2 || c.ws > p2p_ar::kMaxPeers || c.base < 0 ||
      c.base > devices || c.ws > devices - c.base)
    throw std::runtime_error("invalid device range");
  // All pairwise enables complete serially before allocations on ANY rank.
  for (int rank = 0; rank < c.ws; ++rank) {
    check(hipSetDevice(c.base + rank), "peer device");
    for (int peer = 0; peer < c.ws; ++peer) {
      if (rank == peer) continue;
      int can = 0;
      check(hipDeviceCanAccessPeer(&can, c.base + rank, c.base + peer), "peer query");
      if (!can) throw Skip("requested device range lacks all-to-all peer access");
      const auto e = hipDeviceEnablePeerAccess(c.base + peer, 0);
      if (e != hipErrorPeerAccessAlreadyEnabled) check(e, "peer enable");
    }
  }
  for (int rank = 0; rank < c.ws; ++rank) {
    check(hipSetDevice(c.base + rank), "property device");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, c.base + rank), "properties");
    std::fprintf(stderr, "device=%d arch=%s CUs=%d\n", c.base + rank,
                  prop.gcnArchName, prop.multiProcessorCount);
  }
  if (c.blocks == 0) {
    c.blocks = std::min(220, int((c.bytes / 16 + c.threads - 1) / c.threads));
  }
  if (c.op->fixed_blocks) {
    c.blocks = c.op->fixed_blocks;
    c.threads = c.op->fixed_threads;
  }
  // The fill control measures only the common producer, not a gather grid.
  if (c.op->launch == p2p_ar::producer_only) {
    c.blocks = std::min(220, int((c.bytes / 16 + 255) / 256));
    c.threads = 256;
  }
  const int check_slots = std::min(4, c.slots);
  if (c.bytes > std::numeric_limits<size_t>::max() / size_t(c.slots))
    throw std::runtime_error("slab size overflow");
  const size_t slab_bytes = c.bytes * size_t(c.slots);
  Ranks ranks;
  std::vector<float*> peers;
  std::vector<uint64_t*> flags;
  for (int rank = 0; rank < c.ws; ++rank) {
    auto r = std::make_unique<Rank>();
    r->device = c.base + rank;
    check(hipSetDevice(r->device), "allocate device");
    size_t free = 0, total = 0;
    check(hipMemGetInfo(&free, &total), "memory capacity");
    const size_t out_bytes = c.bytes * size_t(check_slots);
    if (slab_bytes > free || out_bytes > free - slab_bytes ||
        (64u << 20) > free - slab_bytes - out_bytes)
      throw std::runtime_error("slab does not fit; reduce slots (capacity, not kernel fallback)");
    check(hipStreamCreateWithFlags(&r->stream, hipStreamNonBlocking), "stream");
    check(hipMalloc(&r->input, slab_bytes), "input slab");
    check(hipMalloc(&r->out, c.bytes * size_t(check_slots)), "separate output");
    check(hipExtMallocWithFlags(reinterpret_cast<void**>(&r->flag), sizeof(uint64_t),
                                hipDeviceMallocFinegrained), "fine-grained flag");
    check(hipMalloc(&r->epoch_base, sizeof(uint64_t)), "epoch parameter");
    check(hipMalloc(&r->peers, size_t(c.ws) * sizeof(float*)), "peer table");
    check(hipMalloc(&r->flags, size_t(c.ws) * sizeof(uint64_t*)), "flag table");
    check(hipMemset(r->flag, 0, sizeof(uint64_t)), "initial flag");
    // Time on each device's own clock. Disable event system fences: the event
    // must not pre-flush data or repair a broken release/acquire protocol.
    check(hipEventCreateWithFlags(&r->start, hipEventDisableSystemFence), "start event");
    check(hipEventCreateWithFlags(&r->stop, hipEventDisableSystemFence), "stop event");
    peers.push_back(r->input); flags.push_back(r->flag);
    ranks.push_back(std::move(r));
  }
  for (int rank = 0; rank < c.ws; ++rank) {
    auto& r = *ranks[rank];
    check(hipSetDevice(r.device), "setup device");
    check(hipMemcpy(r.peers, peers.data(), size_t(c.ws) * sizeof(float*), hipMemcpyHostToDevice), "peer upload");
    check(hipMemcpy(r.flags, flags.data(), size_t(c.ws) * sizeof(uint64_t*), hipMemcpyHostToDevice), "flag upload");
    if (c.mode == "graph") {
      r.validation = capture(r, c, rank, check_slots, true);
      r.timed = capture(r, c, rank, c.slots, false);
    }
  }
  uint64_t base = 0;
  // Every element/slot/rank is checked through 8 changing-input replays.
  for (int iteration = 0; iteration < 8; ++iteration) {
    (void)run(ranks, c, base, check_slots, true);
    verify(ranks, c, base, check_slots, true);
    base += check_slots;
  }
  for (int warm = 0; warm < 3; ++warm) {
    (void)run(ranks, c, base, c.slots, false);
    base += c.slots;
  }
  std::vector<double> gpu, wall;
  for (int sample = 0; sample < c.samples; ++sample) {
    const auto t = run(ranks, c, base, c.slots, false);
    gpu.push_back(t.gpu); wall.push_back(t.wall);
    base += c.slots;
  }
  // Last timed result / input must still be correct after the hot runs.
  verify(ranks, c, base - c.slots, c.slots, false);
  auto sorted = gpu;
  std::sort(sorted.begin(), sorted.end());
  const size_t p90 = size_t(std::ceil(0.9 * sorted.size())) - 1;
  const double us = median(gpu);
  const double output_gbs = double(c.ws) * c.bytes / (us * 1000);
  std::puts("backend,mode,ws,bytes,blocks,threads,slots,samples,gpu_median_us,gpu_min_us,gpu_p90_us,wall_median_us,output_GBs,correct");
  std::printf("%s,%s,%d,%zu,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,OK\n",
               c.op->name, c.mode.c_str(), c.ws, c.bytes, c.blocks, c.threads,
               c.slots, c.samples,
               us, sorted.front(), sorted[p90], median(wall), output_gbs);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 10) {
    std::fprintf(stderr, "usage: %s operation ws bytes [blocks=0] [threads=256] [slots=64] [samples=5] [device_base=0] [graph|eager]\n"
                        "operations: fused pipe chunk hoisted phase peer-major fill\n"
                        "            prefetch2 prefetch4 hoist-prefetch2 peer-prefetch2 peer-prefetch4\n"
                        "            peer-reg1 peer-reg4 peer-reg8 peer-reg32\n", argv[0]);
    return 1;
  }
  try {
    Config c;
    c.op = &operation(argv[1]);
    c.ws = integer(argv[2]);
    const auto bytes = number(argv[3]);
    if (bytes <= 0 || bytes % 16 || uint64_t(bytes / 4) > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("bytes must be positive, divisible by 16, and fit uint32 float indexing");
    c.bytes = size_t(bytes);
    if (argc > 4) c.blocks = integer(argv[4]);
    if (argc > 5) c.threads = integer(argv[5]);
    if (argc > 6) c.slots = integer(argv[6]);
    if (argc > 7) c.samples = integer(argv[7]);
    if (argc > 8) c.base = integer(argv[8]);
    if (argc > 9) c.mode = argv[9];
    if (c.blocks < 0 || c.slots <= 0 || c.samples <= 0 ||
        (c.threads != 64 && c.threads != 128 && c.threads != 256 && c.threads != 512 && c.threads != 1024) ||
        (c.mode != "graph" && c.mode != "eager"))
      throw std::runtime_error("invalid geometry / slots / samples / mode");
    study(c);
  } catch (const Skip& e) {
    std::fprintf(stderr, "SKIP: %s\n", e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
