// P2P allreduce study (research/p2p-allreduce/). Standalone harness: N ranks
// (one thread per GCD), each fills its io row (simulated forward) then runs
// an allreduce. Sweeps backend / world_size / message size / iters to compare
// cross-GCD synchronization strategies for the engine's TP decode path.
//
// Usage: ./study <backend> <world_size> <bytes> <iters> [device_base]
//   backend: hostsync | event | rccl
//     hostsync — hipMemcpyPeer gather/scatter + hipStreamSynchronize + host
//                atomic barrier (the engine's current hand-rolled impl).
//     event    — same data movement, but hipEventCreateWithFlags(
//                hipEventReleaseToSystem|hipEventDisableTiming) +
//                hipStreamWaitEvent for the cross-device dependency. NO host
//                stream sync, NO host blocking — the GPU-side stream dep
//                replaces it. A host atomic only orders record-before-wait
//                (dodges the stale-event re-recording race); it does not block
//                on the GPU.
//     rccl     — ncclAllReduce (baseline; needs the AMD-overlay RCCL on
//                LD_LIBRARY_PATH — Canonical's build segfaults >=2 ranks).
//
// Reports avg us/allreduce, aggregate GB/s (world_size*bytes/time), and a
// correctness check (each rank's io == sum of all ranks' values).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "kernels.hpp"

#if defined(USE_RCCL)
#include <rccl/rccl.h>
#endif

namespace {

#define HIP_CHECK(x)                                                         \
  do {                                                                       \
    const hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s (line %d)\n", #x,                  \
                   hipGetErrorString(e_), __LINE__);                         \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

using Clock = std::chrono::steady_clock;
double us_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

struct State {
  std::string backend;
  int world_size = 1;
  size_t bytes = 0;
  int iters = 0;
  int device_base = 0;

  std::vector<float*> io;          // io[r]: device buffer on GCD (device_base+r)
  float* scratch = nullptr;        // rank 0 only: world_size slots of (bytes/4)
  size_t slot_n = 0;               // floats per slot

  // event-backend shared state
  std::vector<hipEvent_t> in_ev;   // per rank (ReleaseToSystem)
  hipEvent_t out_ev = nullptr;
  // kernelp2p-backend shared state
  std::vector<hipEvent_t> fwd_ev;   // per rank: forward (fill) done
  std::vector<hipEvent_t> rd_ev;    // per rank: flat_read done
  std::vector<float*> scratch_r;    // per-rank local scratch (n floats)
  std::vector<float*> io2;          // per-rank out-of-place dst (kernelp2p_oo / io_b for oo_pipe)
  std::vector<float**> d_peer_io;   // per-rank device copy of the io-ptr array
  std::vector<float**> d_peer_io_b; // per-rank device copy of the io2-ptr array (oo_pipe)
  std::vector<hipEvent_t> fwd_ring; // world_size*4 event ring (oo_pipe): [r*4+slot]
  std::vector<hipEvent_t> rd_ring;  // world_size*4 event ring (oo_pipe)
  // host atomics ordering record-before-wait (NOT GPU-blocking)
  std::atomic<uint32_t> in_count{0};
  std::atomic<uint32_t> out_gen{0};
  std::atomic<uint32_t> fwd_count{0};
  std::atomic<uint32_t> rd_count{0};

  // rccl
  ncclComm_t comm = nullptr;
  ncclUniqueId unique_id{};

  // setup / loop barriers
  std::mutex setup_mu;
  std::condition_variable setup_cv;
  int setup_count = 0;
  std::atomic<bool> go{false};
  std::atomic<int> done_count{0};
  std::mutex done_mu;
  std::condition_variable done_cv;

  // result
  std::vector<double> rank_us;
  std::vector<float> host_io;      // rank 0's io after the loop (correctness)
};

void allreduce_hostsync(State& s, int r, int g) {
  const size_t bytes = s.bytes;
  HIP_CHECK(hipStreamSynchronize(0));  // forward done -> io in HBM
  s.in_count.fetch_add(1, std::memory_order_release);
  while (s.in_count.load(std::memory_order_acquire) <
         uint32_t(s.world_size) * uint32_t(g)) {
  }
  if (r == 0) {
    for (int j = 0; j < s.world_size; ++j)
      HIP_CHECK(hipMemcpyPeerAsync(s.scratch + size_t(j) * s.slot_n, 0, s.io[j],
                                   j, bytes, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    p2p_ar::sum_reduce(s.scratch, s.slot_n, s.io[0], uint32_t(s.slot_n),
                       uint32_t(s.world_size));
    for (int j = 1; j < s.world_size; ++j)
      HIP_CHECK(hipMemcpyPeerAsync(s.io[j], j, s.io[0], 0, bytes, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    s.out_gen.store(uint32_t(g), std::memory_order_release);
  } else {
    while (s.out_gen.load(std::memory_order_acquire) < uint32_t(g)) {
    }
  }
}

void allreduce_event(State& s, int r, int g) {
  // rank r records its "input ready" event (forward is queued on stream 0);
  // ReleaseToSystem flushes this rank's coarse-grained io writes so peers see
  // them once they wait on this event.
  HIP_CHECK(hipEventRecord(s.in_ev[r], 0));
  s.in_count.fetch_add(1, std::memory_order_release);
  if (r == 0) {
    // host: wait until every rank has ISSUED its event record (does NOT wait
    // for the GPU recording — hipStreamWaitEvent below does that). This only
    // orders record-before-wait to dodge the stale re-recording race.
    while (s.in_count.load(std::memory_order_acquire) <
           uint32_t(s.world_size) * uint32_t(g)) {
    }
    // GPU-side: rank 0's stream waits for every rank's forward (and its
    // ReleaseToSystem flush) before gathering. No host blocking.
    for (int j = 0; j < s.world_size; ++j)
      HIP_CHECK(hipStreamWaitEvent(0, s.in_ev[j], 0));
    for (int j = 0; j < s.world_size; ++j)
      HIP_CHECK(hipMemcpyPeerAsync(s.scratch + size_t(j) * s.slot_n, 0, s.io[j],
                                   j, s.bytes, 0));
    p2p_ar::sum_reduce(s.scratch, s.slot_n, s.io[0], uint32_t(s.slot_n),
                       uint32_t(s.world_size));
    for (int j = 1; j < s.world_size; ++j)
      HIP_CHECK(hipMemcpyPeerAsync(s.io[j], j, s.io[0], 0, s.bytes, 0));
    HIP_CHECK(hipEventRecord(s.out_ev, 0));  // scatter done + system-visible
    s.out_gen.store(uint32_t(g), std::memory_order_release);
  } else {
    while (s.out_gen.load(std::memory_order_acquire) < uint32_t(g)) {
    }
    HIP_CHECK(hipStreamWaitEvent(0, s.out_ev, 0));
  }
}

// Flat P2P allreduce: every rank reads ALL peers' io (kernel remote loads,
// the 134 GB/s SM path) and writes its OWN io (local — no cross-GCD write, so
// no remote-write L2 hazard). Two cross-GCD barriers via events: inputs-ready
// (all forwards done before any read) and read-done (all reads done before any
// write — the read→write ordering the flat structure needs to avoid rank r
// writing io[r] while rank k reads it). io is fine-grained-allocated so the
// peer reads are coherent without consumer-L2 staleness across allreduces.
void allreduce_kernelp2p(State& s, int r, int g) {
  const uint32_t n = uint32_t(s.bytes / sizeof(float));
  const uint32_t ws = uint32_t(s.world_size);
  // (the fill/forward is already queued on stream 0 by the loop before this)
  HIP_CHECK(hipEventRecord(s.fwd_ev[r], 0));
  s.fwd_count.fetch_add(1, std::memory_order_release);
  while (s.fwd_count.load(std::memory_order_acquire) < ws * uint32_t(g)) {
  }
  for (int j = 0; j < s.world_size; ++j)
    HIP_CHECK(hipStreamWaitEvent(0, s.fwd_ev[j], 0));  // inputs ready
  p2p_ar::flat_read(s.scratch_r[r], s.d_peer_io[r], n, ws);
  HIP_CHECK(hipEventRecord(s.rd_ev[r], 0));
  s.rd_count.fetch_add(1, std::memory_order_release);
  while (s.rd_count.load(std::memory_order_acquire) < ws * uint32_t(g)) {
  }
  for (int j = 0; j < s.world_size; ++j)
    HIP_CHECK(hipStreamWaitEvent(0, s.rd_ev[j], 0));  // read→write barrier
  p2p_ar::flat_write(s.io[r], s.scratch_r[r], n);
}

// Out-of-place flat allreduce: read all peers' io (src) -> write OWN io2 (dst).
// No in-place read→write race within the kernel (writes go to io2, unread), so
// read+write fuse into one kernel. BUT the read→write dependency across
// blocks remains: the next fill overwrites io (the src), so a reads-complete
// barrier (all allreduce kernels done) still gates the next fill. Same TWO
// barriers as kernelp2p — this variant tests whether fusing the kernel (one
// fewer launch/allreduce) helps. Prediction: marginal (per the README).
void allreduce_kernelp2p_oo(State& s, int r, int g) {
  const uint32_t n = uint32_t(s.bytes / sizeof(float));
  const uint32_t ws = uint32_t(s.world_size);
  HIP_CHECK(hipEventRecord(s.fwd_ev[r], 0));
  s.fwd_count.fetch_add(1, std::memory_order_release);
  while (s.fwd_count.load(std::memory_order_acquire) < ws * uint32_t(g)) {
  }
  for (int j = 0; j < s.world_size; ++j)
    HIP_CHECK(hipStreamWaitEvent(0, s.fwd_ev[j], 0));  // inputs ready
  p2p_ar::flat_read_write(s.io2[r], s.d_peer_io[r], n, ws);  // read io, write io2
  HIP_CHECK(hipEventRecord(s.rd_ev[r], 0));
  s.rd_count.fetch_add(1, std::memory_order_release);
  while (s.rd_count.load(std::memory_order_acquire) < ws * uint32_t(g)) {
  }
  for (int j = 0; j < s.world_size; ++j)
    HIP_CHECK(hipStreamWaitEvent(0, s.rd_ev[j], 0));  // reads-complete -> next fill
}

// Pipelined out-of-place flat allreduce: double-buffered io (io/io2 alternate
// as src/dst), so the next-but-one forward reuses a buffer rather than the
// next one. The reads-complete barrier is DEFERRED two iterations (lag-2):
// fill(g) waits for allreduce(g-2) done before overwriting its src — by then
// the allreduce(g-2) kernel is long finished, so the wait is non-blocking
// ("free"). Only the inputs-ready barrier still blocks per iteration. This
// is the test of whether pipelining recovers the barrier cost the flat
// structure pays. Uses 4-slot event rings (pipeline depth 2 < 4) to avoid the
// single-event re-recording race/over-serialization. NB: the harness's
// "forward" is a trivial fill, so this understates the win vs the real engine
// where block.forward compute hides the deferred barrier.
void allreduce_kernelp2p_oo_pipe(State& s, int r, int g) {
  const uint32_t n = uint32_t(s.bytes / sizeof(float));
  const uint32_t ws = uint32_t(s.world_size);
  const int slot = (g - 1) & 3;
  const int prev = (g - 2) & 3;
  const bool odd = ((g - 1) & 1);
  float* src = odd ? s.io2[r] : s.io[r];            // io_((g-1)mod2)
  float* dst = odd ? s.io[r] : s.io2[r];            // the other buffer
  float* const* dpsrc = odd ? s.d_peer_io_b[r] : s.d_peer_io[r];

  // lag-2 reads-complete: allreduce(g-2) done before fill(g) overwrites src.
  if (g >= 3) {
    while (s.rd_count.load(std::memory_order_acquire) < ws * uint32_t(g - 2)) {
    }
    for (int j = 0; j < ws; ++j)
      HIP_CHECK(hipStreamWaitEvent(0, s.rd_ring[j * 4 + prev], 0));
  }
  // forward (fill) — done here (not in the loop) because the src buffer alternates
  p2p_ar::fill_value(src, n, float(r + 1) + float(g) * 1e-3f);
  HIP_CHECK(hipEventRecord(s.fwd_ring[r * 4 + slot], 0));
  s.fwd_count.fetch_add(1, std::memory_order_release);
  while (s.fwd_count.load(std::memory_order_acquire) < ws * uint32_t(g)) {
  }
  for (int j = 0; j < ws; ++j)
    HIP_CHECK(hipStreamWaitEvent(0, s.fwd_ring[j * 4 + slot], 0));  // inputs ready
  p2p_ar::flat_read_write(dst, dpsrc, n, ws);
  HIP_CHECK(hipEventRecord(s.rd_ring[r * 4 + slot], 0));
  s.rd_count.fetch_add(1, std::memory_order_release);
}

#if defined(USE_RCCL)
void allreduce_rccl(State& s, int r, int /*g*/) {
  ncclResult_t nr = ncclAllReduce(s.io[r], s.io[r], s.slot_n, ncclFloat,
                                  ncclSum, s.comm, 0);
  if (nr != ncclSuccess) {
    std::fprintf(stderr, "rank %d ncclAllReduce failed: %s\n", r,
                 ncclGetErrorString(nr));
    std::exit(1);
  }
}
#endif

void rank_thread(State& s, int r) {
  // Context creation must be serial across rank threads (concurrent first-touch
  // HIP context creation segfaults libamdhip64, and it destabilizes RCCL init
  // — the fix used by research/rccl-single-process). hipSetDevice is per-thread
  // and sticky, so the device stays this rank's after the lock releases.
  {
    static std::mutex ctx_mu;
    std::lock_guard<std::mutex> lock(ctx_mu);
    HIP_CHECK(hipSetDevice(s.device_base + r));
    void* probe = nullptr;
    HIP_CHECK(hipMalloc(&probe, 16));
    HIP_CHECK(hipFree(probe));
  }
  HIP_CHECK(hipSetDevice(s.device_base + r));
  const uint32_t n = uint32_t(s.bytes / sizeof(float));

  // io: fine-grained for kernelp2p (coherent cross-GCD peer reads), coarse
  // (plain hipMalloc) otherwise. KP_IO_COARSE=1 forces coarse for kernelp2p —
  // a probe of whether fine-grained is actually needed (it bypasses/coheres the
  // L2; coarse relies on the writer's hipEventReleaseToSystem flush + the
  // reader NOT stale-caching peer reads).
  const bool want_fg = (s.backend == "kernelp2p" || s.backend == "kernelp2p_oo" ||
                        s.backend == "kernelp2p_oo_pipe") && !getenv("KP_IO_COARSE");
  if (want_fg)
    HIP_CHECK(hipExtMallocWithFlags((void**)&s.io[r], s.bytes, hipDeviceMallocFinegrained));
  else
    HIP_CHECK(hipMalloc(&s.io[r], s.bytes));
  p2p_ar::fill_value(s.io[r], n, float(r + 1));  // initial fill
  if (r == 0) {
    s.slot_n = n;
    HIP_CHECK(hipMalloc(&s.scratch, s.bytes * size_t(s.world_size)));
  }
  if (s.backend == "event") {
    HIP_CHECK(hipEventCreateWithFlags(&s.in_ev[r],
                                      hipEventDisableTiming | hipEventReleaseToSystem));
    if (r == 0)
      HIP_CHECK(hipEventCreateWithFlags(&s.out_ev,
                                        hipEventDisableTiming | hipEventReleaseToSystem));
  }
  if (s.backend == "kernelp2p" || s.backend == "kernelp2p_oo" ||
      s.backend == "kernelp2p_oo_pipe") {
    HIP_CHECK(hipMalloc(&s.scratch_r[r], s.bytes));          // local read scratch (kernelp2p only)
    HIP_CHECK(hipMalloc((void**)&s.d_peer_io[r],
                        size_t(s.world_size) * sizeof(float*)));
    HIP_CHECK(hipMalloc(&s.io2[r], s.bytes));                // out-of-place dst / io_b
    if (s.backend == "kernelp2p_oo_pipe")
      HIP_CHECK(hipMalloc((void**)&s.d_peer_io_b[r],
                          size_t(s.world_size) * sizeof(float*)));
    HIP_CHECK(hipEventCreateWithFlags(&s.fwd_ev[r],
                                      hipEventDisableTiming | hipEventReleaseToSystem));
    HIP_CHECK(hipEventCreateWithFlags(&s.rd_ev[r],
                                      hipEventDisableTiming | hipEventReleaseToSystem));
    if (s.backend == "kernelp2p_oo_pipe") {
      for (int sl = 0; sl < 4; ++sl) {
        hipEvent_t e;
        HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming | hipEventReleaseToSystem));
        s.fwd_ring[r * 4 + sl] = e;
        HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming | hipEventReleaseToSystem));
        s.rd_ring[r * 4 + sl] = e;
      }
    }
  }
#if defined(USE_RCCL)
  if (s.backend == "rccl") {
    ncclResult_t nr =
        ncclCommInitRank(&s.comm, s.world_size, s.unique_id, r);
    if (nr != ncclSuccess) {
      std::fprintf(stderr, "rank %d ncclCommInitRank failed: %s\n", r,
                   ncclGetErrorString(nr));
      std::exit(1);
    }
  }
#endif

  // setup barrier
  {
    std::lock_guard<std::mutex> lock(s.setup_mu);
    if (++s.setup_count == s.world_size) s.setup_cv.notify_all();
  }

  // wait for go (host barrier: all ranks ready before the timed loop)
  {
    std::unique_lock<std::mutex> lock(s.setup_mu);
    s.setup_cv.wait(lock, [&] { return s.go.load(); });
  }

  // kernelp2p / kernelp2p_oo: enable peer access serially (after ALL allocs —
  // enabling it racing concurrent hipMalloc trips an HSA MappedHandleAllowedAgent
  // assert, cf. the engine), then build the per-rank device copy of the io-ptr array.
  if (s.backend == "kernelp2p" || s.backend == "kernelp2p_oo" ||
      s.backend == "kernelp2p_oo_pipe") {
    {
      static std::mutex peer_mu;
      std::lock_guard<std::mutex> lock(peer_mu);
      for (int p = 0; p < s.world_size; ++p) {
        if (p == r) continue;
        int can = 0;
        if (hipDeviceCanAccessPeer(&can, r, p) != hipSuccess || !can) continue;
        const hipError_t e = hipDeviceEnablePeerAccess(p, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
          std::fprintf(stderr, "rank %d peer access to %d failed: %s\n", r, p,
                       hipGetErrorString(e));
          std::exit(1);
        }
      }
    }
    HIP_CHECK(hipMemcpyHtoD(s.d_peer_io[r], s.io.data(),
                            size_t(s.world_size) * sizeof(float*)));
    if (s.backend == "kernelp2p_oo_pipe")
      HIP_CHECK(hipMemcpyHtoD(s.d_peer_io_b[r], s.io2.data(),
                              size_t(s.world_size) * sizeof(float*)));
  }

  const auto t0 = Clock::now();
  for (int g = 1; g <= s.iters; ++g) {
    // simulated forward: refill io[r] (oo_pipe does its own alternating fill)
    if (s.backend != "kernelp2p_oo_pipe")
      p2p_ar::fill_value(s.io[r], n, float(r + 1) + float(g) * 1e-3f);
    if (s.backend == "hostsync") allreduce_hostsync(s, r, g);
    else if (s.backend == "event") allreduce_event(s, r, g);
    else if (s.backend == "kernelp2p") allreduce_kernelp2p(s, r, g);
    else if (s.backend == "kernelp2p_oo") allreduce_kernelp2p_oo(s, r, g);
    else if (s.backend == "kernelp2p_oo_pipe") allreduce_kernelp2p_oo_pipe(s, r, g);
#if defined(USE_RCCL)
    else if (s.backend == "rccl") allreduce_rccl(s, r, g);
#endif
    else { std::fprintf(stderr, "unknown backend\n"); std::exit(1); }
  }
  HIP_CHECK(hipStreamSynchronize(0));
  const auto t1 = Clock::now();
  s.rank_us[r] = us_between(t0, t1);

  // signal done
  {
    std::lock_guard<std::mutex> lock(s.done_mu);
    if (++s.done_count == s.world_size) s.done_cv.notify_all();
  }

  // correctness: rank 0 reads back the result buffer (io for in-place,
  // io2 for kernelp2p_oo, the last iteration's dst for oo_pipe).
  if (r == 0) {
    s.host_io.assign(n, 0.0f);
    const float* res;
    if (s.backend == "kernelp2p_oo") res = s.io2[0];
    else if (s.backend == "kernelp2p_oo_pipe") res = (((s.iters - 1) & 1) ? s.io[0] : s.io2[0]);
    else res = s.io[0];
    HIP_CHECK(hipMemcpy(s.host_io.data(), res, s.bytes, hipMemcpyDeviceToHost));
  }

  // cleanup
  if (s.backend == "event") {
    if (s.in_ev[r]) hipEventDestroy(s.in_ev[r]);
    if (r == 0 && s.out_ev) hipEventDestroy(s.out_ev);
  }
  if (s.backend == "kernelp2p" || s.backend == "kernelp2p_oo" ||
      s.backend == "kernelp2p_oo_pipe") {
    if (s.fwd_ev[r]) hipEventDestroy(s.fwd_ev[r]);
    if (s.rd_ev[r]) hipEventDestroy(s.rd_ev[r]);
    if (s.scratch_r[r]) hipFree(s.scratch_r[r]);
    if (s.io2[r]) hipFree(s.io2[r]);
    if (s.d_peer_io[r]) hipFree(s.d_peer_io[r]);
    if (s.d_peer_io_b[r]) hipFree(s.d_peer_io_b[r]);
    if (s.backend == "kernelp2p_oo_pipe")
      for (int sl = 0; sl < 4; ++sl) {
        hipEventDestroy(s.fwd_ring[r * 4 + sl]);
        hipEventDestroy(s.rd_ring[r * 4 + sl]);
      }
  }
#if defined(USE_RCCL)
  if (s.backend == "rccl" && s.comm) ncclCommDestroy(s.comm);
#endif
  hipFree(s.io[r]);
  if (r == 0 && s.scratch) hipFree(s.scratch);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <backend> <world_size> <bytes> <iters> [device_base]\n"
                 "  backend: hostsync | event | kernelp2p | kernelp2p_oo | kernelp2p_oo_pipe | rccl\n",
                 argv[0]);
    return 1;
  }
  State s;
  s.backend = argv[1];
  s.world_size = std::atoi(argv[2]);
  s.bytes = size_t(std::atoll(argv[3]));
  s.iters = std::atoi(argv[4]);
  s.device_base = argc > 5 ? std::atoi(argv[5]) : 0;
  if (s.bytes % sizeof(float) != 0 || s.bytes < sizeof(float)) {
    std::fprintf(stderr, "bytes must be a positive multiple of 4\n");
    return 1;
  }

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (s.world_size < 2 || s.world_size > ndev) {
    std::fprintf(stderr, "world_size %d out of range (devices=%d)\n", s.world_size, ndev);
    return 1;
  }

#if defined(USE_RCCL)
  if (s.backend == "rccl") {
    ncclResult_t nr = ncclGetUniqueId(&s.unique_id);
    if (nr != ncclSuccess) {
      std::fprintf(stderr, "ncclGetUniqueId failed: %s\n", ncclGetErrorString(nr));
      return 1;
    }
  }
#endif

  s.io.assign(s.world_size, nullptr);
  s.in_ev.assign(s.world_size, nullptr);
  s.fwd_ev.assign(s.world_size, nullptr);
  s.rd_ev.assign(s.world_size, nullptr);
  s.scratch_r.assign(s.world_size, nullptr);
  s.io2.assign(s.world_size, nullptr);
  s.d_peer_io.assign(s.world_size, nullptr);
  s.d_peer_io_b.assign(s.world_size, nullptr);
  s.fwd_ring.assign(s.world_size * 4, nullptr);
  s.rd_ring.assign(s.world_size * 4, nullptr);
  s.rank_us.assign(s.world_size, 0.0);

  std::vector<std::thread> threads;
  for (int r = 0; r < s.world_size; ++r)
    threads.emplace_back(rank_thread, std::ref(s), r);

  // wait for setup
  {
    std::unique_lock<std::mutex> lock(s.setup_mu);
    s.setup_cv.wait(lock, [&] { return s.setup_count == s.world_size; });
  }
  // release all ranks into the timed loop together
  {
    std::lock_guard<std::mutex> lock(s.setup_mu);
    s.go.store(true);
  }
  s.setup_cv.notify_all();

  // wait for completion
  {
    std::unique_lock<std::mutex> lock(s.done_mu);
    s.done_cv.wait(lock, [&] { return s.done_count == s.world_size; });
  }
  for (auto& t : threads) t.join();

  // wall = max rank time; per-iter = wall/iters
  double wall = 0;
  for (double v : s.rank_us) wall = std::max(wall, v);
  const double per_iter = wall / s.iters;
  const double gb_s =
      (double(s.world_size) * double(s.bytes)) / (per_iter * 1e3);  // B/us = GB/s

  // correctness: expected sum of last-gen values
  const uint32_t n = uint32_t(s.bytes / sizeof(float));
  float expect = 0.0f;
  for (int r = 0; r < s.world_size; ++r)
    expect += float(r + 1) + float(s.iters) * 1e-3f;
  bool ok = true;
  for (uint32_t i = 0; i < n; ++i)
    if (s.host_io[i] != expect) { ok = false; break; }

  std::printf("%-9s ws=%d bytes=%-9zu iters=%d  %.2f us/iter  %.1f GB/s  %s\n",
              s.backend.c_str(), s.world_size, s.bytes, s.iters, per_iter, gb_s,
              ok ? "OK" : "FAIL(sum)");
  return ok ? 0 : 2;
}
