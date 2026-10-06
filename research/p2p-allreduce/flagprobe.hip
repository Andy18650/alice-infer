// flagprobe.hip — device-side cross-GCD flag protocol: fine-grained device VRAM
// flags (primary) vs host-pinned flags (reference, to verify the NUMA/PCIe
// objection: GPUs 0-3 are on NUMA node 0, GPUs 4-7 on node 1, so a host-pinned
// page is local-node for half the GCDs and crosses the CPU-side fabric for the
// other half; fine-grained device flags keep flag traffic on XGMI).
//
// Protocol (from atomics.md, probe-verified):
//   writer: payload writes; RELEASE/SYSTEM store to its own flag
//   reader: RELAXED/SYSTEM glc-poll peers' flags; ONE ACQUIRE/SYSTEM load
//           (buffer_invl2) after all arrive; then plain local data loads
//
// Phases:
//   solo     — per-rank poll latency to rank0's flag (bimodal in host mode)
//   pingpong — round-trip latency rank0 <-> rank j
//   barrier  — pure ws-rank device-side barrier, per-barrier latency
//   allreduce— write-scatter allreduce at n floats/rank: peer writes -> flag
//              protocol -> local reads; verifies the l2probe hazard is fixed
//              (remote write -> local read goes FRESH after the acquire) and
//              measures per-allreduce latency (compare: host-event flat
//              ~100 us, RCCL ~45.7 us from README).
//
// Usage: ./flagprobe [mode=fine|host] [ws] [iters] [n_floats]
// Run host mode under `numactl --cpunodebind=0` to pin the flag page to node 0.
#include <hip/hip_runtime.h>
#include <rccl.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

__device__ inline void flag_release(unsigned int* f, unsigned int v) {
  __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline unsigned int flag_poll(const unsigned int* f) {
  return __hip_atomic_load((unsigned int*)f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void flag_acquire(const unsigned int* f) {
  (void)__hip_atomic_load((unsigned int*)f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ---- solo: poll flags[0] `iters` times, report avg latency in ns (wall 25MHz)
__global__ void k_solo(unsigned int* const* flags, int iters, unsigned long long* out) {
  if (threadIdx.x != 0) return;
  unsigned int s = 0;
  unsigned long long t0 = wall_clock64();
  for (int i = 0; i < iters; ++i) s += flag_poll(flags[0]);
  unsigned long long t1 = wall_clock64();
  out[0] = (t1 - t0);  // ticks; *40 = ns
  if (s == 0xdeadbeef) out[1] = 1;  // keep alive
}

// ---- pingpong: rank0 is leader, partner follows; round trips
__global__ void k_pingpong(unsigned int* my_flag, unsigned int* peer_flag,
                           int iters, int leader) {
  if (threadIdx.x != 0) return;
  for (int e = 1; e <= iters; ++e) {
    if (leader) {
      flag_release(my_flag, (unsigned)e);
      while (flag_poll(peer_flag) < (unsigned)e) {}
      flag_acquire(my_flag);
    } else {
      while (flag_poll(peer_flag) < (unsigned)e) {}
      flag_acquire(my_flag);
      flag_release(my_flag, (unsigned)e);
    }
  }
}

// ---- pure ws-rank barrier, `iters` barriers
__global__ void k_barrier(unsigned int* my_flag, unsigned int* const* flags,
                          int ws, int rank, int iters) {
  if (threadIdx.x != 0) return;
  for (int e = 1; e <= iters; ++e) {
    flag_release(my_flag, (unsigned)e);
    for (int j = 0; j < ws; ++j)
      if (j != rank) while (flag_poll(flags[j]) < (unsigned)e) {}
    flag_acquire(my_flag);
  }
}

// ---- write-scatter allreduce with two device-side barriers per epoch
__global__ void k_allreduce(float* my_payload, float* const* payloads,
                            unsigned int* my_flag, unsigned int* const* flags,
                            float* out, unsigned int* mism,
                            int ws, int rank, unsigned int n, int iters) {
  for (int e = 1; e <= iters; ++e) {
    int eterm = ((e - 1) & 63) * 1024;
    // write my contribution into slot[rank] of EVERY rank's payload
    for (int j = 0; j < ws; ++j) {
      float* dst = payloads[j] + (size_t)rank * n;
      for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = (float)(rank * 65536 + eterm + (int)(i & 1023));
    }
    __syncthreads();
    // produce barrier
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e - 1));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e - 1)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
    // read all slots locally, sum, verify
    float base = 0;
    for (int j = 0; j < ws; ++j) base += (float)(j * 65536 + eterm);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
      float s = 0;
      for (int j = 0; j < ws; ++j) s += my_payload[(size_t)j * n + i];
      out[i] = s;
      if (s != base + (float)(ws * (int)(i & 1023))) atomicAdd(mism, 1u);
    }
    __syncthreads();
    // consume barrier (peers may not overwrite until all have read)
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
  }
}

// ---- read-gather allreduce (the user's proposal): sync -> every rank reads
// every other rank's contribution buffer REMOTELY, reducing on the fly in
// registers -> sync -> result stored locally. Producers only ever write
// locally (cheap); consumers do all the remote traffic as READS (measured
// fast path, ~135 GB/s). No slot scratch needed. Same flag protocol.
__global__ void k_allreduce_rg(float* my_contrib, float* out,
                               float* const* contribs,
                               unsigned int* my_flag, unsigned int* const* flags,
                               unsigned int* mism,
                               int ws, int rank, unsigned int n, int iters) {
  for (int e = 1; e <= iters; ++e) {
    int eterm = ((e - 1) & 63) * 1024;
    // local fill of my contribution (simulates the forward write)
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
      my_contrib[i] = (float)(rank * 65536 + eterm + (int)(i & 1023));
    __syncthreads();
    // produce barrier
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e - 1));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e - 1)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
    // read-gather: sum all ranks' contribs (peers remote, mine local)
    float base = 0;
    for (int j = 0; j < ws; ++j) base += (float)(j * 65536 + eterm);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
      float s = 0;
      for (int j = 0; j < ws; ++j) s += contribs[j][i];
      out[i] = s;
      if (s != base + (float)(ws * (int)(i & 1023))) atomicAdd(mism, 1u);
    }
    __syncthreads();
    // consume barrier (peers may not refill their contrib until all have read)
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
  }
}

// ---- read-gather, float4-vectorized remote reads
__global__ void k_allreduce_rg4(float* my_contrib, float* out,
                                float* const* contribs,
                                unsigned int* my_flag, unsigned int* const* flags,
                                unsigned int* mism,
                                int ws, int rank, unsigned int n, int iters) {
  unsigned int n4 = n / 4;
  for (int e = 1; e <= iters; ++e) {
    int eterm = ((e - 1) & 63) * 1024;
    float4* mc = (float4*)my_contrib;
    for (unsigned int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
      unsigned int i = 4 * i4;
      float b = (float)(rank * 65536 + eterm);
      mc[i4] = make_float4(b + (float)(i & 1023), b + (float)((i + 1) & 1023),
                           b + (float)((i + 2) & 1023), b + (float)((i + 3) & 1023));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e - 1));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e - 1)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
    float base = 0;
    for (int j = 0; j < ws; ++j) base += (float)(j * 65536 + eterm);
    float4* o4 = (float4*)out;
    for (unsigned int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int j = 0; j < ws; ++j) {
        float4 v = ((const float4*)contribs[j])[i4];
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      o4[i4] = s;
      unsigned int i = 4 * i4;
      float e0 = base + (float)(ws * (int)(i & 1023));
      float e1 = base + (float)(ws * (int)((i + 1) & 1023));
      float e2 = base + (float)(ws * (int)((i + 2) & 1023));
      float e3 = base + (float)(ws * (int)((i + 3) & 1023));
      if (s.x != e0 || s.y != e1 || s.z != e2 || s.w != e3) atomicAdd(mism, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
  }
}

// ---- write-scatter, float4 (wide block)
__global__ void k_allreduce_ws4(float* my_payload, float* const* payloads,
                                unsigned int* my_flag, unsigned int* const* flags,
                                float* out, unsigned int* mism,
                                int ws, int rank, unsigned int n, int iters) {
  unsigned int n4 = n / 4;
  for (int e = 1; e <= iters; ++e) {
    int eterm = ((e - 1) & 63) * 1024;
    float b = (float)(rank * 65536 + eterm);
    for (int j = 0; j < ws; ++j) {
      float4* dst = (float4*)(payloads[j] + (size_t)rank * n);
      for (unsigned int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
        unsigned int i = 4 * i4;
        dst[i4] = make_float4(b + (float)(i & 1023), b + (float)((i + 1) & 1023),
                              b + (float)((i + 2) & 1023), b + (float)((i + 3) & 1023));
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e - 1));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e - 1)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
    float base = 0;
    for (int j = 0; j < ws; ++j) base += (float)(j * 65536 + eterm);
    float4* o4 = (float4*)out;
    for (unsigned int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int j = 0; j < ws; ++j) {
        float4 v = ((const float4*)my_payload)[(size_t)j * n / 4 + i4];
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      o4[i4] = s;
      unsigned int i = 4 * i4;
      if (s.x != base + (float)(ws * (int)(i & 1023)) ||
          s.y != base + (float)(ws * (int)((i + 1) & 1023)) ||
          s.z != base + (float)(ws * (int)((i + 2) & 1023)) ||
          s.w != base + (float)(ws * (int)((i + 3) & 1023))) atomicAdd(mism, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      flag_release(my_flag, (unsigned)(2 * e));
      for (int j = 0; j < ws; ++j)
        if (j != rank) while (flag_poll(flags[j]) < (unsigned)(2 * e)) {}
      flag_acquire(my_flag);
    }
    __syncthreads();
  }
}

// ---- multi-block pipeline read-gather: separate fill / barrier / gather
// kernels per epoch (stream ordering gives the local deps; the 1-block flag
// kernel does the cross-GCD release/poll/acquire; the acquire's buffer_invl2
// covers the L2 shared by all CUs, and kernel-launch resets L1s, so the wide
// gather kernel reads fresh). This is the bandwidth-regime structure.
__global__ void k_fill4(float* my_contrib, int rank, int eterm, unsigned int n) {
  unsigned int n4 = n / 4;
  float4* mc = (float4*)my_contrib;
  float b = (float)(rank * 65536 + eterm);
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += stride) {
    unsigned int i = 4 * i4;
    mc[i4] = make_float4(b + (float)(i & 1023), b + (float)((i + 1) & 1023),
                         b + (float)((i + 2) & 1023), b + (float)((i + 3) & 1023));
  }
}
__global__ void k_flagbar(unsigned int* my_flag, unsigned int* const* flags,
                          int ws, int rank, unsigned int value) {
  if (threadIdx.x != 0) return;
  flag_release(my_flag, value);
  for (int j = 0; j < ws; ++j)
    if (j != rank) while (flag_poll(flags[j]) < value) {}
  flag_acquire(my_flag);
}
__global__ void k_gather4(float* const* contribs, float* out, unsigned int* mism,
                          int ws, int eterm, unsigned int n) {
  unsigned int n4 = n / 4;
  float base = 0;
  for (int j = 0; j < ws; ++j) base += (float)(j * 65536 + eterm);
  float4* o4 = (float4*)out;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += stride) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < ws; ++j) {
      float4 v = ((const float4*)contribs[j])[i4];
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    o4[i4] = s;
    unsigned int i = 4 * i4;
    if (s.x != base + (float)(ws * (int)(i & 1023)) ||
        s.y != base + (float)(ws * (int)((i + 1) & 1023)) ||
        s.z != base + (float)(ws * (int)((i + 2) & 1023)) ||
        s.w != base + (float)(ws * (int)((i + 3) & 1023))) atomicAdd(mism, 1u);
  }
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "fine";
  int ws     = argc > 2 ? atoi(argv[2]) : 4;
  int iters  = argc > 3 ? atoi(argv[3]) : 300;
  unsigned int n = argc > 4 ? (unsigned int)atoi(argv[4]) : 8192;
  bool fine = strcmp(mode, "fine") == 0;
  printf("mode=%s ws=%d iters=%d n=%u (%u KB/rank)\n", mode, ws, iters, n, n * 4 / 1024);

  std::vector<float*> payload(ws), out(ws);
  std::vector<unsigned int*> mism(ws);
  std::vector<unsigned long long*> solo_t(ws);
  std::vector<unsigned int*> flag_mem(ws);
  unsigned int* host_flags = nullptr;

  for (int r = 0; r < ws; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&payload[r], (size_t)ws * n * 4));
    HIP_CHECK(hipMalloc(&out[r], (size_t)n * 4));
    HIP_CHECK(hipMalloc(&mism[r], 4));
    HIP_CHECK(hipMalloc(&solo_t[r], 16));
    HIP_CHECK(hipMemset(mism[r], 0, 4));
    if (fine)
      HIP_CHECK(hipExtMallocWithFlags((void**)&flag_mem[r], 4, hipDeviceMallocFinegrained));
  }
  if (!fine) {
    HIP_CHECK(hipHostMalloc((void**)&host_flags, (size_t)ws * 4));
    memset(host_flags, 0, (size_t)ws * 4);
    for (int r = 0; r < ws; ++r) flag_mem[r] = host_flags + r;
  } else {
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
    }
  }
  // peer access, serial, after all allocations (concurrent enable + hipMalloc
  // trips an HSA assert — see README)
  for (int r = 0; r < ws; ++r) {
    HIP_CHECK(hipSetDevice(r));
    for (int j = 0; j < ws; ++j) {
      if (j == r) continue;
      int can = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can, r, j));
      if (can) HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
    }
  }
  // per-rank device pointer tables
  std::vector<float*> h_pay(ws);
  std::vector<unsigned int*> h_flag(ws);
  for (int r = 0; r < ws; ++r) { h_pay[r] = payload[r]; h_flag[r] = flag_mem[r]; }
  std::vector<float**> d_pay(ws);
  std::vector<unsigned int**> d_flag(ws);
  for (int r = 0; r < ws; ++r) {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMalloc(&d_pay[r], ws * sizeof(float*)));
    HIP_CHECK(hipMalloc(&d_flag[r], ws * sizeof(unsigned int*)));
    HIP_CHECK(hipMemcpy(d_pay[r], h_pay.data(), ws * sizeof(float*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_flag[r], h_flag.data(), ws * sizeof(unsigned int*), hipMemcpyHostToDevice));
  }
  for (int r = 0; r < ws; ++r) { HIP_CHECK(hipSetDevice(r)); HIP_CHECK(hipDeviceSynchronize()); }

  auto run_all = [&](auto kern) {
    std::vector<std::thread> ts;
    std::vector<double> ms(ws, 0.0);
    for (int r = 0; r < ws; ++r)
      ts.emplace_back([&, r] {
        HIP_CHECK(hipSetDevice(r));
        auto t0 = std::chrono::steady_clock::now();
        kern(r);
        HIP_CHECK(hipDeviceSynchronize());
        ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      });
    for (auto& t : ts) t.join();
    double mx = 0; for (double v : ms) mx = v > mx ? v : mx;
    return mx;
  };

  // ---- solo poll latency to rank0's flag
  run_all([&](int r) { k_solo<<<1, 32>>>(d_flag[r], 20000, solo_t[r]); });
  printf("solo poll of rank0 flag (ns/load):");
  for (int r = 0; r < ws; ++r) {
    unsigned long long ticks = 0;
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipMemcpy(&ticks, solo_t[r], 8, hipMemcpyDeviceToHost));
    printf("  r%d=%.0f", r, ticks * 40.0 / 20000.0);
  }
  printf("\n");

  // ---- pingpong rank0 <-> j
  for (int j = 1; j < ws; ++j) {
    std::thread t0([&] {
      HIP_CHECK(hipSetDevice(0));
      auto t = std::chrono::steady_clock::now();
      k_pingpong<<<1, 32>>>(flag_mem[0], flag_mem[j], iters, 1);
      HIP_CHECK(hipDeviceSynchronize());
      double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
      printf("pingpong 0<->%d: %.2f us/round-trip\n", j, us / iters);
    });
    std::thread tj([&] {
      HIP_CHECK(hipSetDevice(j));
      k_pingpong<<<1, 32>>>(flag_mem[j], flag_mem[0], iters, 0);
      HIP_CHECK(hipDeviceSynchronize());
    });
    t0.join(); tj.join();
    // reset flags for next pair
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      if (fine) HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
    }
    if (!fine) memset(host_flags, 0, (size_t)ws * 4);
    for (int r = 0; r < ws; ++r) { HIP_CHECK(hipSetDevice(r)); HIP_CHECK(hipDeviceSynchronize()); }
  }

  // ---- pure barrier
  {
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      if (fine) HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
    }
    if (!fine) memset(host_flags, 0, (size_t)ws * 4);
    double ms = run_all([&](int r) { k_barrier<<<1, 32>>>(flag_mem[r], d_flag[r], ws, r, iters); });
    printf("barrier ws=%d: %.2f us/barrier\n", ws, ms * 1000.0 / iters);
  }

  // ---- write-scatter allreduce
  {
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      if (fine) HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
    }
    if (!fine) memset(host_flags, 0, (size_t)ws * 4);
    double ms = run_all([&](int r) {
      k_allreduce<<<1, 256>>>(payload[r], d_pay[r], flag_mem[r], d_flag[r],
                              out[r], mism[r], ws, r, n, iters);
    });
    unsigned int total_mism = 0;
    for (int r = 0; r < ws; ++r) {
      unsigned int m = 0;
      HIP_CHECK(hipSetDevice(r));
      HIP_CHECK(hipMemcpy(&m, mism[r], 4, hipMemcpyDeviceToHost));
      total_mism += m;
    }
    printf("allreduce(write-scatter) ws=%d: %.2f us/allreduce  mismatches=%u -> %s\n",
           ws, ms * 1000.0 / iters, total_mism, total_mism == 0 ? "FRESH/OK" : "STALE/FAIL");
  }

  // ---- read-gather allreduce (scalar + float4)
  for (int variant = 0; variant < 2; ++variant) {
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      if (fine) HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
      HIP_CHECK(hipMemset(mism[r], 0, 4));
    }
    if (!fine) memset(host_flags, 0, (size_t)ws * 4);
    double ms = run_all([&](int r) {
      if (variant == 0)
        k_allreduce_rg<<<1, 256>>>(payload[r], out[r], d_pay[r], flag_mem[r], d_flag[r],
                                   mism[r], ws, r, n, iters);
      else
        k_allreduce_rg4<<<1, 1024>>>(payload[r], out[r], d_pay[r], flag_mem[r], d_flag[r],
                                    mism[r], ws, r, n, iters);
    });
    unsigned int total_mism = 0;
    for (int r = 0; r < ws; ++r) {
      unsigned int m = 0;
      HIP_CHECK(hipSetDevice(r));
      HIP_CHECK(hipMemcpy(&m, mism[r], 4, hipMemcpyDeviceToHost));
      total_mism += m;
    }
    printf("allreduce(read-gather%s) ws=%d: %.2f us/allreduce  mismatches=%u -> %s\n",
           variant == 0 ? "" : "-f4", ws, ms * 1000.0 / iters, total_mism,
           total_mism == 0 ? "FRESH/OK" : "STALE/FAIL");
  }

  // ---- write-scatter f4, 1024 threads
  {
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      if (fine) HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
      HIP_CHECK(hipMemset(mism[r], 0, 4));
    }
    if (!fine) memset(host_flags, 0, (size_t)ws * 4);
    double ms = run_all([&](int r) {
      k_allreduce_ws4<<<1, 1024>>>(payload[r], d_pay[r], flag_mem[r], d_flag[r],
                                   out[r], mism[r], ws, r, n, iters);
    });
    unsigned int total_mism = 0;
    for (int r = 0; r < ws; ++r) {
      unsigned int m = 0;
      HIP_CHECK(hipSetDevice(r));
      HIP_CHECK(hipMemcpy(&m, mism[r], 4, hipMemcpyDeviceToHost));
      total_mism += m;
    }
    printf("allreduce(write-scatter-f4) ws=%d: %.2f us/allreduce  mismatches=%u -> %s\n",
           ws, ms * 1000.0 / iters, total_mism, total_mism == 0 ? "FRESH/OK" : "STALE/FAIL");
  }

  // ---- multi-block pipeline read-gather (fill/bar/gather/bar per epoch)
  {
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      if (fine) HIP_CHECK(hipMemset(flag_mem[r], 0, 4));
      HIP_CHECK(hipMemset(mism[r], 0, 4));
    }
    if (!fine) memset(host_flags, 0, (size_t)ws * 4);
    unsigned int n4 = n / 4;
    int grid = (int)((n4 + 255) / 256); if (grid > 64) grid = 64; if (grid < 1) grid = 1;
    double ms = run_all([&](int r) {
      for (int e = 1; e <= iters; ++e) {
        int eterm = ((e - 1) & 63) * 1024;
        k_fill4<<<grid, 256>>>(payload[r], r, eterm, n);
        k_flagbar<<<1, 32>>>(flag_mem[r], d_flag[r], ws, r, (unsigned)(2 * e - 1));
        k_gather4<<<grid, 256>>>(d_pay[r], out[r], mism[r], ws, eterm, n);
        k_flagbar<<<1, 32>>>(flag_mem[r], d_flag[r], ws, r, (unsigned)(2 * e));
      }
    });
    unsigned int total_mism = 0;
    for (int r = 0; r < ws; ++r) {
      unsigned int m = 0;
      HIP_CHECK(hipSetDevice(r));
      HIP_CHECK(hipMemcpy(&m, mism[r], 4, hipMemcpyDeviceToHost));
      total_mism += m;
    }
    printf("allreduce(rg-f4-pipe, %d blocks) ws=%d: %.2f us/allreduce  mismatches=%u -> %s\n",
           grid, ws, ms * 1000.0 / iters, total_mism, total_mism == 0 ? "FRESH/OK" : "STALE/FAIL");
  }

  // ---- RCCL reference (AMD-overlay librccl via LD_LIBRARY_PATH at runtime)
  {
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) { fprintf(stderr, "ncclGetUniqueId failed\n"); return 1; }
    std::vector<ncclComm_t> comms(ws, nullptr);
    std::vector<std::thread> ts;
    for (int r = 0; r < ws; ++r)
      ts.emplace_back([&, r] {
        HIP_CHECK(hipSetDevice(r));
        if (ncclCommInitRank(&comms[r], ws, id, r) != ncclSuccess) {
          fprintf(stderr, "ncclCommInitRank failed r%d\n", r); exit(1);
        }
      });
    for (auto& t : ts) t.join();
    // correctness (integer-valued data, exact in fp32 under any sum order)
    std::vector<float> h(n);
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      for (unsigned int i = 0; i < n; ++i) h[i] = (float)(r * 65536 + (int)(i & 1023));
      HIP_CHECK(hipMemcpy(payload[r], h.data(), (size_t)n * 4, hipMemcpyHostToDevice));
    }
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      ncclAllReduce(payload[r], payload[r], n, ncclFloat, ncclSum, comms[r], 0);
    }
    for (int r = 0; r < ws; ++r) { HIP_CHECK(hipSetDevice(r)); HIP_CHECK(hipDeviceSynchronize()); }
    unsigned int bad = 0;
    for (int r = 0; r < ws; ++r) {
      HIP_CHECK(hipSetDevice(r));
      HIP_CHECK(hipMemcpy(h.data(), payload[r], (size_t)n * 4, hipMemcpyDeviceToHost));
      double base = (double)ws * (ws - 1) / 2.0 * 65536.0;
      for (unsigned int i = 0; i < n; ++i)
        if ((double)h[i] != base + (double)ws * (double)(i & 1023)) ++bad;
    }
    // (a) sync-per-call latency (the TP decode critical path), (b) pipelined
    std::vector<double> lat(ws, 0), pipe(ws, 0);
    std::vector<std::thread> ts2;
    for (int r = 0; r < ws; ++r)
      ts2.emplace_back([&, r] {
        HIP_CHECK(hipSetDevice(r));
        auto t0 = std::chrono::steady_clock::now();
        for (int e = 0; e < iters; ++e) {
          ncclAllReduce(payload[r], payload[r], n, ncclFloat, ncclSum, comms[r], 0);
          HIP_CHECK(hipStreamSynchronize(0));
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int e = 0; e < iters; ++e)
          ncclAllReduce(payload[r], payload[r], n, ncclFloat, ncclSum, comms[r], 0);
        HIP_CHECK(hipDeviceSynchronize());
        auto t2 = std::chrono::steady_clock::now();
        lat[r] = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        pipe[r] = std::chrono::duration<double, std::micro>(t2 - t1).count() / iters;
      });
    for (auto& t : ts2) t.join();
    double mlat = 0, mpipe = 0;
    for (int r = 0; r < ws; ++r) { mlat = std::max(mlat, lat[r]); mpipe = std::max(mpipe, pipe[r]); }
    printf("rccl ws=%d: %.2f us sync-per-call, %.2f us pipelined  bad=%u -> %s\n",
           ws, mlat, mpipe, bad, bad == 0 ? "exact" : "FAIL");
    for (int r = 0; r < ws; ++r) { HIP_CHECK(hipSetDevice(r)); ncclCommDestroy(comms[r]); }
  }
  return 0;
}
