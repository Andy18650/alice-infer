// Optional out-of-place RCCL baseline. Shares only the study's GPU producer;
// no production dependency and no changes to the P2P operation dispatcher.
// One communicator set is reused for all payloads and both launch modes.
#include "kernels.hpp"
#include <rccl/rccl.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef RCCL_EXPECTED_LIBRARY
#define RCCL_EXPECTED_LIBRARY ""
#endif

namespace {
void hip_check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void nccl_check(ncclResult_t e, const char* what) {
  if (e != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(e));
}
void verify_library() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<const void*>(&ncclAllReduce), &info) || !info.dli_fname)
    throw std::runtime_error("cannot identify loaded RCCL library");
  if (std::string(RCCL_EXPECTED_LIBRARY).empty() ||
      std::filesystem::canonical(info.dli_fname) != std::filesystem::canonical(RCCL_EXPECTED_LIBRARY))
    throw std::runtime_error("refusing unverified RCCL: " + std::string(info.dli_fname) +
                             "; set LD_LIBRARY_PATH to the documented AMD 7.1 overlay");
  int version = 0;
  nccl_check(ncclGetVersion(&version), "RCCL version");
  std::fprintf(stderr, "RCCL version=%d library=%s\n", version,
                std::filesystem::canonical(info.dli_fname).c_str());
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
  float *input = nullptr, *output = nullptr;
  uint64_t *epoch = nullptr, host_epoch = 0;
  ncclComm_t comm = nullptr;
  std::unique_ptr<Graph> validation, timed;
  ~Rank() {
    (void)hipSetDevice(device);
    // On an exception abort any outstanding communication before releasing
    // captured graphs/buffers. Normal completion destroys comms explicitly.
    if (comm) (void)ncclCommAbort(comm);
    validation.reset(); timed.reset();
    if (start) (void)hipEventDestroy(start);
    if (stop) (void)hipEventDestroy(stop);
    if (stream) (void)hipStreamDestroy(stream);
    if (input) (void)hipFree(input);
    if (output) (void)hipFree(output);
    if (epoch) (void)hipFree(epoch);
  }
};
using Ranks = std::vector<std::unique_ptr<Rank>>;

void enqueue_all(Ranks& ranks, size_t bytes, int slots, bool validation) {
  // One group per collective: producer -> collective remains ordered on each
  // stream, unlike grouping the whole batch and queuing all producers first.
  for (int slot = 0; slot < slots; ++slot) {
    nccl_check(ncclGroupStart(), "group start");
    try {
      for (int rank = 0; rank < int(ranks.size()); ++rank) {
        auto& r = *ranks[rank];
        hip_check(hipSetDevice(r.device), "enqueue device");
        p2p_ar::Args a;
        a.input = r.input; a.out = r.output; a.epoch_base = r.epoch;
        a.n = uint32_t(bytes / 4); a.offset = size_t(slot) * a.n;
        a.rank = rank; a.ws = int(ranks.size()); a.ordinal = uint64_t(slot + 1);
        if (validation) a.out += a.offset;
        p2p_ar::fill(r.stream, a);
        nccl_check(ncclAllReduce(a.input + a.offset, a.out, a.n, ncclFloat,
                                 ncclSum, r.comm, r.stream), "out-of-place allreduce");
      }
    } catch (...) {
      (void)ncclGroupEnd();
      throw;
    }
    nccl_check(ncclGroupEnd(), "group end");
  }
}
void capture_all(Ranks& ranks, size_t bytes, int slots, bool validation) {
  // All ranks must be in capture before grouped collective submission.
  // Capture records, not executes; no rank is warmed alone in a collective.
  size_t begun = 0, ended = 0;
  try {
    for (auto& r : ranks) {
      hip_check(hipSetDevice(r->device), "capture device");
      auto graph = std::make_unique<Graph>();
      if (validation) r->validation = std::move(graph);
      else r->timed = std::move(graph);
      hip_check(hipStreamBeginCapture(r->stream, hipStreamCaptureModeThreadLocal), "begin capture");
      ++begun;
    }
    enqueue_all(ranks, bytes, slots, validation);
    for (auto& r : ranks) {
      hip_check(hipSetDevice(r->device), "instantiate device");
      auto& g = validation ? r->validation : r->timed;
      const auto e = hipStreamEndCapture(r->stream, &g->graph);
      ++ended;
      hip_check(e, "end capture");
      hip_check(hipGraphInstantiate(&g->exec, g->graph, nullptr, nullptr, 0), "instantiate");
    }
  } catch (...) {
    for (size_t i = ended; i < begun; ++i) {
      auto& r = ranks[i];
      (void)hipSetDevice(r->device);
      auto& g = validation ? r->validation : r->timed;
      (void)hipStreamEndCapture(r->stream, &g->graph);
    }
    throw;
  }
}
struct Times { double gpu = 0, wall = 0; };
Times run(Ranks& ranks, size_t bytes, uint64_t base, int slots, bool validation, bool graph) {
  for (auto& r : ranks) {
    hip_check(hipSetDevice(r->device), "prepare device");
    r->host_epoch = base;
    hip_check(hipMemcpyAsync(r->epoch, &r->host_epoch, sizeof(base), hipMemcpyHostToDevice,
                             r->stream), "epoch upload");
    if (validation)
      hip_check(hipMemsetAsync(r->output, 0xff, bytes * size_t(slots), r->stream), "poison output");
    hip_check(hipEventRecord(r->start, r->stream), "start event");
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (graph) {
    for (auto& r : ranks) {
      hip_check(hipSetDevice(r->device), "launch device");
      hip_check(hipGraphLaunch((validation ? r->validation : r->timed)->exec, r->stream), "graph launch");
    }
  } else enqueue_all(ranks, bytes, slots, validation);
  // GroupEnd must have issued every NCCL operation before stop events are
  // recorded. Never place stop events inside an open NCCL group.
  for (auto& r : ranks) {
    hip_check(hipSetDevice(r->device), "stop device");
    hip_check(hipEventRecord(r->stop, r->stream), "stop event");
  }
  for (auto& r : ranks) {
    hip_check(hipSetDevice(r->device), "wait device");
    hip_check(hipStreamSynchronize(r->stream), "completion");
  }
  Times t;
  t.wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / slots;
  for (auto& r : ranks) {
    float ms = 0;
    hip_check(hipSetDevice(r->device), "event device");
    hip_check(hipEventElapsedTime(&ms, r->start, r->stop), "event elapsed");
    t.gpu = std::max(t.gpu, double(ms) * 1000 / slots);
  }
  return t;
}
void verify(Ranks& ranks, size_t bytes, uint64_t base, int slots, bool all) {
  const size_t n = bytes / 4, count = all ? size_t(slots) * n : n;
  std::vector<float> input(count), output(count);
  for (int rank = 0; rank < int(ranks.size()); ++rank) {
    auto& r = *ranks[rank];
    hip_check(hipSetDevice(r.device), "verify device");
    hip_check(hipMemcpy(input.data(), r.input + (all ? 0 : size_t(slots - 1) * n),
                        count * 4, hipMemcpyDeviceToHost), "input download");
    hip_check(hipMemcpy(output.data(), r.output, count * 4, hipMemcpyDeviceToHost), "output download");
    for (int s = 0; s < (all ? slots : 1); ++s) {
      const int slot = all ? s : slots - 1;
      const uint64_t common = ((base + slot + 1) % 1024) * 4;
      const uint64_t sum_base = uint64_t(128) * ranks.size() * (ranks.size() + 1) / 2 + ranks.size() * common;
      for (size_t i = 0; i < n; ++i) {
        const float own = float((rank + 1) * 128 + common + i % 31) * 0.03125f;
        const float sum = float(sum_base + ranks.size() * (i % 31)) * 0.03125f;
        if (input[size_t(s) * n + i] != own || output[size_t(s) * n + i] != sum)
          throw std::runtime_error("incorrect RCCL sum or overwritten input at rank " +
                                   std::to_string(rank) + " slot " + std::to_string(slot));
      }
    }
  }
}
double median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  return x.size() % 2 ? x[x.size() / 2] : (x[x.size() / 2 - 1] + x[x.size() / 2]) / 2;
}
void measure(Ranks& ranks, size_t bytes, int slots, int samples, bool graph, uint64_t& base) {
  const int checked = std::min(slots, 4);
  for (auto& r : ranks) {
    hip_check(hipSetDevice(r->device), "clear graphs device");
    r->validation.reset(); r->timed.reset();
  }
  // Warm the communicator/algorithm before capture. Group all ranks and wait
  // only after every rank is submitted. Outside all timing/validation samples.
  (void)run(ranks, bytes, base, 1, true, false); base += 1;
  if (graph) {
    capture_all(ranks, bytes, checked, true);
    capture_all(ranks, bytes, slots, false);
  }
  for (int i = 0; i < 8; ++i) {
    (void)run(ranks, bytes, base, checked, true, graph);
    verify(ranks, bytes, base, checked, true);
    base += checked;
  }
  for (int i = 0; i < 3; ++i) {
    (void)run(ranks, bytes, base, slots, false, graph); base += slots;
  }
  std::vector<double> gpu, wall;
  for (int i = 0; i < samples; ++i) {
    const auto t = run(ranks, bytes, base, slots, false, graph);
    gpu.push_back(t.gpu); wall.push_back(t.wall); base += slots;
  }
  verify(ranks, bytes, base - slots, slots, false);
  auto sorted = gpu; std::sort(sorted.begin(), sorted.end());
  std::printf("rccl,%s,%zu,%zu,0,0,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,OK\n",
               graph ? "graph" : "eager", ranks.size(), bytes, slots, samples,
               median(gpu), sorted.front(), sorted[size_t(std::ceil(.9 * sorted.size())) - 1],
               median(wall), double(ranks.size()) * bytes / (median(gpu) * 1000));
  std::fflush(stdout);
}
int integer(const char* text) {
  size_t used = 0; const int value = std::stoi(text, &used);
  if (text[used] != '\0' || value < 0) throw std::runtime_error("invalid argument");
  return value;
}
void study(int ws, int slots, int samples, int device_base, const std::string& mode) {
  verify_library();  // refuse Canonical's broken build before any collective
  int devices = 0;
  hip_check(hipGetDeviceCount(&devices), "device count");
  if (ws < 2 || ws > 8 || device_base > devices || ws > devices - device_base || slots < 1 || samples < 1)
    throw std::runtime_error("invalid ranks/slots/samples/device range");
  for (int r = 0; r < ws; ++r) {
    hip_check(hipSetDevice(device_base + r), "peer device");
    for (int p = 0; p < ws; ++p) {
      if (p == r) continue;
      int can = 0; hip_check(hipDeviceCanAccessPeer(&can, device_base + r, device_base + p), "peer query");
      if (!can) throw std::runtime_error("required peer access unavailable");
      const auto e = hipDeviceEnablePeerAccess(device_base + p, 0);
      if (e != hipErrorPeerAccessAlreadyEnabled) hip_check(e, "pre-allocation peer enable");
    }
  }
  constexpr size_t max_bytes = 4u << 20;  // no 16 MiB search
  Ranks ranks;
  for (int r = 0; r < ws; ++r) {
    auto rank = std::make_unique<Rank>(); rank->device = device_base + r;
    hip_check(hipSetDevice(rank->device), "allocate device");
    hip_check(hipStreamCreateWithFlags(&rank->stream, hipStreamNonBlocking), "stream");
    hip_check(hipMalloc(&rank->input, max_bytes * size_t(slots)), "immutable input slab");
    hip_check(hipMalloc(&rank->output, max_bytes * size_t(std::min(slots, 4))), "separate output");
    hip_check(hipMalloc(&rank->epoch, sizeof(uint64_t)), "epoch control");
    hip_check(hipEventCreateWithFlags(&rank->start, hipEventDisableSystemFence), "start event");
    hip_check(hipEventCreateWithFlags(&rank->stop, hipEventDisableSystemFence), "stop event");
    ranks.push_back(std::move(rank));
  }
  std::vector<int> devlist; std::vector<ncclComm_t> comms(ws, nullptr);
  for (auto& r : ranks) devlist.push_back(r->device);
  const auto t0 = std::chrono::steady_clock::now();
  const auto init = ncclCommInitAll(comms.data(), ws, devlist.data());
  for (int r = 0; r < ws; ++r) ranks[r]->comm = comms[r];
  nccl_check(init, "single-process communicator init");
  std::fprintf(stderr, "communicators initialized in %.3f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  std::puts("backend,mode,ws,bytes,blocks,threads,slots,samples,gpu_median_us,gpu_min_us,gpu_p90_us,wall_median_us,output_GBs,correct");
  uint64_t base = 0;
  for (size_t bytes : {size_t(32768), size_t(1048576), max_bytes}) {
    if (mode != "eager") measure(ranks, bytes, slots, samples, true, base);
    if (mode != "graph") measure(ranks, bytes, slots, samples, false, base);
  }
  // Destroy all graphs before communicator resources. GPU work is globally
  // complete; normal communicator destruction is local, not an abort.
  for (auto& r : ranks) {
    hip_check(hipSetDevice(r->device), "graph cleanup device");
    r->validation.reset(); r->timed.reset();
  }
  for (auto& r : ranks) {
    hip_check(hipSetDevice(r->device), "comm cleanup device");
    nccl_check(ncclCommDestroy(r->comm), "comm destroy"); r->comm = nullptr;
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 6) {
    std::fprintf(stderr, "usage: %s ws [slots=64] [samples=9] [device_base=0] [graph|eager|both]\n", argv[0]);
    return 1;
  }
  try {
    const std::string mode = argc > 5 ? argv[5] : "both";
    if (mode != "graph" && mode != "eager" && mode != "both") throw std::runtime_error("invalid mode");
    study(integer(argv[1]), argc > 2 ? integer(argv[2]) : 64,
          argc > 3 ? integer(argv[3]) : 9, argc > 4 ? integer(argv[4]) : 0, mode);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
  }
}
