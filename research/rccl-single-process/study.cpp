// Single-process multi-rank RCCL viability study (alice-infer research).
//
// Question: can ONE process own N in-process RCCL ranks (one per GCD) and
// get correct, low-latency allreduces at TP-relevant message sizes? If yes,
// tensor parallelism needs no SPMD/multi-process machinery.
//
// Host-only C++ (no hipcc): RCCL + HIP C APIs, like the earlier scout
// benches. One persistent std::thread per rank: hipSetDevice, comm init,
// correctness allreduce, latency sweep. Build: ./build.sh

#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define HIP_CHECK(x)                                                   \
  do {                                                                 \
    hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                            \
      std::fprintf(stderr, "rank %d: %s failed: %s (line %d)\n", rank, \
                   #x, hipGetErrorString(e_), __LINE__);               \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

#define NCCL_CHECK(x)                                                  \
  do {                                                                 \
    ncclResult_t e_ = (x);                                             \
    if (e_ != ncclSuccess) {                                           \
      std::fprintf(stderr, "rank %d: %s failed: %s (line %d)\n", rank, \
                   #x, ncclGetErrorString(e_), __LINE__);              \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

constexpr size_t kMaxBytes = 16 << 20;  // prefill chunk: 512 x 8192 x fp32
constexpr size_t kMaxCount = kMaxBytes / sizeof(float);

struct RankCtx {
  hipStream_t stream = nullptr;
  float* buf = nullptr;
  ncclComm_t comm = nullptr;
};

struct Shared {
  int nranks;
  ncclUniqueId id;
  std::barrier<> bar;            // rank rendezvous for timed sections
  std::vector<double> init_ms;   // per-rank init wall time
  std::vector<RankCtx> rc;
  explicit Shared(int n) : nranks(n), bar(n), init_ms(n, 0.0), rc(n) {}
};

void rank_main(int rank, Shared& sh) {
  // HIP setup was done serially by the owner thread, but the *current
  // device* is per-thread state: bind this thread to its rank's context
  // (RCCL validates the calling thread's device against the rank).
  HIP_CHECK(hipSetDevice(rank));
  RankCtx& rc = sh.rc[rank];
  hipStream_t stream = rc.stream;
  float* buf = rc.buf;

  // Phase A: collective init from this thread; must overlap other ranks.
  const auto t0 = std::chrono::steady_clock::now();
  NCCL_CHECK(ncclCommInitRank(&rc.comm, sh.nranks, sh.id, rank));
  ncclComm_t comm = rc.comm;
  sh.init_ms[rank] =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
          .count();

  // Phase B: correctness at full size (in-place, mirrors TP stream buffer).
  std::vector<float> host(kMaxCount);
  NCCL_CHECK(ncclAllReduce(buf, buf, kMaxCount, ncclFloat, ncclSum, comm, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(host.data(), buf, kMaxBytes, hipMemcpyDeviceToHost));
  const double base = double(sh.nranks) * (sh.nranks - 1) / 2.0;
  size_t bad = 0;
  for (size_t i = 0; i < kMaxCount; ++i) {
    const double want = base + double(sh.nranks) * double(i % 13);
    if (double(host[i]) != want) ++bad;
  }
  if (rank == 0)
    std::printf("[correct] %d ranks: %s (%zu/%zu wrong)\n", sh.nranks,
                bad ? "FAIL" : "exact match", bad, kMaxCount);

  // Phase C: latency sweep. (a) sync-per-call latency = the TP decode
  // critical-path number; (b) pipelined per-call = throughput reference.
  static const size_t kSizes[] = {8 << 10, 32 << 10, 64 << 10, 256 << 10, 1 << 20,
                                  kMaxBytes};
  for (size_t bytes : kSizes) {
    const size_t count = bytes / sizeof(float);
    const int warm = 20, iters = bytes <= (64 << 10) ? 200 : 50;
    for (int i = 0; i < warm; ++i)
      NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    sh.bar.arrive_and_wait();
    const auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
      HIP_CHECK(hipStreamSynchronize(stream));  // (a) sync per call
    }
    sh.bar.arrive_and_wait();
    const double sync_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0)
            .count() /
        iters;

    const auto p0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
      NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
    HIP_CHECK(hipStreamSynchronize(stream));  // (b) one sync at the end
    sh.bar.arrive_and_wait();
    const double pipe_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - p0)
            .count() /
        iters;

    if (rank == 0)
      std::printf("[latency] %7zu KB: sync/call %8.1f us   pipelined %8.1f us   "
                  "(%.1f GB/s eff.)\n",
                  bytes >> 10, sync_us, pipe_us,
                  2.0 * sh.nranks * (sh.nranks - 1) * bytes / (sh.nranks * pipe_us * 1e3));
  }

  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
}

}  // namespace

int main(int argc, char** argv) {
  const int rank = -1;  // for the CHECK macros outside rank_main
  const int nranks = argc > 1 ? std::atoi(argv[1]) : 8;
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (nranks < 2 || nranks > ndev) {
    std::fprintf(stderr, "nranks %d out of range (devices: %d)\n", nranks, ndev);
    return 1;
  }
  int ver = 0;
  NCCL_CHECK(ncclGetVersion(&ver));
  std::printf("devices=%d ranks=%d rccl version=%d\n", ndev, nranks, ver);

  Shared sh(nranks);
  NCCL_CHECK(ncclGetUniqueId(&sh.id));

  // Serial HIP setup for every rank, owned by the main thread (mirrors the
  // engine: the model object owns all device state; threads only drive
  // collectives). Also serializes HIP's lazy per-device context init.
  std::vector<float> host(kMaxCount);
  for (int r = 0; r < nranks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamCreate(&sh.rc[r].stream));
    HIP_CHECK(hipMalloc(&sh.rc[r].buf, kMaxBytes));
    for (size_t i = 0; i < kMaxCount; ++i)
      host[i] = float(r) + float(i % 13);  // exact in fp32
    HIP_CHECK(hipMemcpy(sh.rc[r].buf, host.data(), kMaxBytes, hipMemcpyHostToDevice));
  }
  std::fprintf(stderr, "[setup] HIP contexts + buffers for %d ranks OK\n", nranks);

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  threads.reserve(nranks);
  for (int r = 0; r < nranks; ++r) threads.emplace_back(rank_main, r, std::ref(sh));
  for (auto& t : threads) t.join();
  for (int r = 0; r < nranks; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipFree(sh.rc[r].buf));
    HIP_CHECK(hipStreamDestroy(sh.rc[r].stream));
  }
  std::printf("[init] all ranks done; wall %.0f ms (per-rank max %.0f ms)\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                  .count(),
              *std::max_element(sh.init_ms.begin(), sh.init_ms.end()));
  return 0;
}
