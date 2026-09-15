// Prefill attention scheme A/B experiment (research harness, NOT engine code).
//
// Single-head-style causal attention kernels over fp32 Q/K/V:
//   query s of S attends keys [0, T-S+s]; scale = 1/sqrt(hd).
//   grid.y = head h (H heads for machine-fill during timing), blockIdx.x =
//   query (v1) or query tile (A, B).
//
//   v1: engine baseline. One block per (query, head), 128 threads,
//       lane-per-element, block reduction per key row, online softmax.
//   A:  streaming single-pass, thread-per-query. q in registers, acc in
//       registers (hd=64) or LDS (hd=128: q+acc can't share the register
//       file), online rescale per key. No chunk structure, no syncthreads.
//   B:  two-phase per 64-key chunk (user's scheme). Raw score tile in LDS
//       during the K phase (q row staged in regs from an LDS q tile),
//       deferred exp(score - chunk_max) in the V phase (acc row staged in
//       regs from an LDS acc tile) — phase-separated registers, one online
//       rescale per chunk. Score rows are thread-private: no per-chunk
//       __syncthreads.
//
// Query-tile sizes are per-scheme per-hd best-occupancy picks (documented in
// README): A64=64, A128=32, B64=32, B128=16.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,  \
                   __LINE__);                                                            \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

namespace {

// ---------------------------------------------------------------- v1 (engine baseline)

template <unsigned NT>
__device__ float blk_sum(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  return shared[0];
}

template <uint32_t HD>
__global__ void k_v1(const float* __restrict__ q, const float* __restrict__ kc,
                     const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                     uint32_t T, float scale) {
  __shared__ float shared[128];
  const uint32_t s = blockIdx.x;
  const float* qh = q + ((size_t)blockIdx.y * S + s) * HD;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;
  const uint32_t Tk = T - S + s + 1;
  const uint32_t i = threadIdx.x;
  const float q_i = i < HD ? qh[i] : 0.0f;
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (uint32_t t = 0; t < Tk; ++t) {
    const float part = i < HD ? q_i * kh[(size_t)t * HD + i] : 0.0f;
    const float dot = blk_sum<128>(part, shared);
    const float sc = dot * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    if (i < HD) acc = acc * f + p * vh[(size_t)t * HD + i];
    m = nm;
  }
  if (i < HD) out[((size_t)blockIdx.y * S + s) * HD + i] = acc / l;
}

// ---------------------------------------------------------------- scheme A (streaming)

template <uint32_t HD, uint32_t QT, bool ACC_LDS>
__global__ void k_A(const float* __restrict__ q, const float* __restrict__ kc,
                    const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                    uint32_t T, float scale) {
  __shared__ float sacc[ACC_LDS ? QT * HD : 1];
  if (threadIdx.x >= QT) return;  // blocks are 64 threads; QT may be smaller
  const uint32_t s = blockIdx.x * QT + threadIdx.x;
  const bool live = s < S;
  const uint32_t Tk = live ? (T - S + s + 1) : 0;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;

  float q_r[HD];
  const float* qrow = q + ((size_t)blockIdx.y * S + (live ? s : 0)) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) q_r[i] = live ? qrow[i] : 0.0f;
  float acc_r[ACC_LDS ? 1 : HD];
  if constexpr (ACC_LDS) {
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) sacc[threadIdx.x * HD + i] = 0.0f;
  } else {
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) acc_r[i] = 0.0f;
  }

  float m = -INFINITY, l = 0.0f;
  for (uint32_t t = 0; t < Tk; ++t) {
    const float* krow = kh + (size_t)t * HD;  // broadcast: all lanes same address
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += q_r[i] * krow[i];
      d1 += q_r[i + 1] * krow[i + 1];
      d2 += q_r[i + 2] * krow[i + 2];
      d3 += q_r[i + 3] * krow[i + 3];
    }
    const float sc = ((d0 + d1) + (d2 + d3)) * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    const float* vrow = vh + (size_t)t * HD;  // broadcast
    if constexpr (ACC_LDS) {
#pragma unroll
      for (uint32_t i = 0; i < HD; ++i)
        sacc[threadIdx.x * HD + i] = sacc[threadIdx.x * HD + i] * f + p * vrow[i];
    } else {
#pragma unroll
      for (uint32_t i = 0; i < HD; ++i) acc_r[i] = acc_r[i] * f + p * vrow[i];
    }
    m = nm;
  }
  if (!live) return;
  float* orow = out + ((size_t)blockIdx.y * S + s) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i)
    orow[i] = (ACC_LDS ? sacc[threadIdx.x * HD + i] : acc_r[i]) / l;
}

// Scheme A2 (bonus variant for hd=128): q tile in LDS, acc in registers.
// Halves the per-key LDS traffic of A's hd=128 form (q reads instead of acc
// read+write); acc 128 regs fits since q is not co-resident.
template <uint32_t HD, uint32_t QT>
__global__ void k_A2(const float* __restrict__ q, const float* __restrict__ kc,
                     const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                     uint32_t T, float scale) {
  __shared__ float sq[QT * HD];
  for (uint32_t idx = threadIdx.x; idx < QT * HD; idx += 64) {
    const uint32_t r = idx / HD, c = idx % HD;
    const uint32_t gs = blockIdx.x * QT + r;
    sq[idx] = gs < S ? q[((size_t)blockIdx.y * S + gs) * HD + c] : 0.0f;
  }
  __syncthreads();  // the only one in the kernel
  if (threadIdx.x >= QT) return;
  const uint32_t s = blockIdx.x * QT + threadIdx.x;
  const bool live = s < S;
  const uint32_t Tk = live ? (T - S + s + 1) : 0;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;

  float acc[HD];
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) acc[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (uint32_t t = 0; t < Tk; ++t) {
    const float* krow = kh + (size_t)t * HD;  // broadcast
    const float* qrow = sq + threadIdx.x * HD;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += qrow[i] * krow[i];
      d1 += qrow[i + 1] * krow[i + 1];
      d2 += qrow[i + 2] * krow[i + 2];
      d3 += qrow[i + 3] * krow[i + 3];
    }
    const float sc = ((d0 + d1) + (d2 + d3)) * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    const float* vrow = vh + (size_t)t * HD;  // broadcast
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) acc[i] = acc[i] * f + p * vrow[i];
    m = nm;
  }
  if (!live) return;
  float* orow = out + ((size_t)blockIdx.y * S + s) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) orow[i] = acc[i] / l;
}

// ---------------------------------------------------------------- scheme B (two-phase)

template <uint32_t HD, uint32_t QT, uint32_t CK>
__global__ void k_B(const float* __restrict__ q, const float* __restrict__ kc,
                    const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                    uint32_t T, float scale) {
  __shared__ float sq[QT * HD];   // q tile, staged once (coalesced)
  __shared__ float ssc[QT * CK];  // raw score tile (thread-private rows)
  __shared__ float sac[QT * HD];  // acc tile (persistent; regs only inside V phase)
  const uint32_t s = blockIdx.x * QT + threadIdx.x;
  const bool live = s < S;
  const uint32_t Tk = live ? (T - S + s + 1) : 0;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;

  for (uint32_t idx = threadIdx.x; idx < QT * HD; idx += 64) {
    const uint32_t r = idx / HD, c = idx % HD;
    const uint32_t gs = blockIdx.x * QT + r;
    sq[idx] = gs < S ? q[((size_t)blockIdx.y * S + gs) * HD + c] : 0.0f;
    sac[idx] = 0.0f;
  }
  __syncthreads();  // the only one in the kernel
  if (threadIdx.x >= QT) return;  // after the barrier: per-thread LDS rows are QT-sized

  float m = -INFINITY, l = 0.0f;
  for (uint32_t c0 = 0; c0 < Tk; c0 += CK) {
    const uint32_t c1 = min(Tk, c0 + CK);
    // K phase: q row into registers, dots -> score tile, track chunk max.
    float qr[HD];
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) qr[i] = sq[threadIdx.x * HD + i];
    float mc = -INFINITY;
    for (uint32_t t = c0; t < c1; ++t) {
      const float* krow = kh + (size_t)t * HD;  // broadcast
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < HD; i += 4) {
        d0 += qr[i] * krow[i];
        d1 += qr[i + 1] * krow[i + 1];
        d2 += qr[i + 2] * krow[i + 2];
        d3 += qr[i + 3] * krow[i + 3];
      }
      const float sc = ((d0 + d1) + (d2 + d3)) * scale;
      ssc[threadIdx.x * CK + (t - c0)] = sc;
      mc = fmaxf(mc, sc);
    }
    // Merge: rescale history into the new max's units (first chunk: f=0).
    const float M = fmaxf(m, mc);
    const float f = expf(m - M);
    l *= f;
    // V phase: acc row into registers, deferred exp, accumulate, store back.
    float ar[HD];
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) ar[i] = sac[threadIdx.x * HD + i] * f;
    for (uint32_t t = c0; t < c1; ++t) {
      const float p = expf(ssc[threadIdx.x * CK + (t - c0)] - M);
      l += p;
      const float* vrow = vh + (size_t)t * HD;  // broadcast
#pragma unroll
      for (uint32_t i = 0; i < HD; ++i) ar[i] += p * vrow[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) sac[threadIdx.x * HD + i] = ar[i];
    m = M;
  }
  if (!live) return;
  float* orow = out + ((size_t)blockIdx.y * S + s) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) orow[i] = sac[threadIdx.x * HD + i] / l;
}

// ---------------------------------------------------------------- host harness

struct Case {
  uint32_t H, S, T, hd;
  std::vector<float> q, k, v, ref;
};

Case load_case(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  char magic[4];
  Case c;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "ATT1", 4) != 0 ||
      std::fread(&c.H, 4, 1, f) != 1 || std::fread(&c.S, 4, 1, f) != 1 ||
      std::fread(&c.T, 4, 1, f) != 1 || std::fread(&c.hd, 4, 1, f) != 1) {
    std::fprintf(stderr, "bad case file %s\n", path);
    std::exit(1);
  }
  auto read = [&](size_t n) {
    std::vector<float> x(n);
    if (std::fread(x.data(), 4, n, f) != n) {
      std::fprintf(stderr, "truncated case file %s\n", path);
      std::exit(1);
    }
    return x;
  };
  c.q = read((size_t)c.H * c.S * c.hd);
  c.k = read((size_t)c.H * c.T * c.hd);
  c.v = read((size_t)c.H * c.T * c.hd);
  c.ref = read((size_t)c.H * c.S * c.hd);
  std::fclose(f);
  return c;
}

using Launch = void (*)(const float*, const float*, const float*, float*, uint32_t,
                        uint32_t, uint32_t, float);

template <uint32_t HD>
void launch_one(const char* scheme, const Case& c, const float* dq, const float* dk,
                const float* dv, float* dout) {
  const float scale = 1.0f / sqrtf(float(HD));
  const dim3 g1(c.S, c.H);
  if (std::strcmp(scheme, "v1") == 0) {
    k_v1<HD><<<g1, 128>>>(dq, dk, dv, dout, c.S, c.T, scale);
  } else if (std::strcmp(scheme, "A") == 0) {
    if constexpr (HD == 64) {
      const dim3 g((c.S + 63) / 64, c.H);
      k_A<64, 64, false><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    } else {
      const dim3 g((c.S + 63) / 64, c.H);  // sacc 32KB -> 2 blocks/CU, full wavefront
      k_A<128, 64, true><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    }
  } else if (std::strcmp(scheme, "A2") == 0) {
    const dim3 g((c.S + 63) / 64, c.H);  // sq 32KB (hd=128) -> 2 blocks/CU
    if constexpr (HD == 64)
      k_A2<64, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    else
      k_A2<128, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
  } else {
    if constexpr (HD == 64) {
      const dim3 g((c.S + 63) / 64, c.H);  // LDS 48KB -> 1 block/CU, full wavefront
      k_B<64, 64, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    } else {
      // LDS 40KB -> 1 block/CU, half wavefront: B's LDS appetite (q+scores+acc)
      // at hd=128 cannot have both full wavefronts and multi-block occupancy.
      const dim3 g((c.S + 31) / 32, c.H);
      k_B<128, 32, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <case.bin> [--bench N]\n", argv[0]);
    return 1;
  }
  const Case c = load_case(argv[1]);
  const bool write_out = c.H == 1;  // outs only needed for correctness cases
  float *dq, *dk, *dv, *dout;
  HIP_CHECK(hipMalloc(&dq, c.q.size() * 4));
  HIP_CHECK(hipMalloc(&dk, c.k.size() * 4));
  HIP_CHECK(hipMalloc(&dv, c.v.size() * 4));
  HIP_CHECK(hipMalloc(&dout, c.q.size() * 4));
  HIP_CHECK(hipMemcpy(dq, c.q.data(), c.q.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dk, c.k.data(), c.k.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dv, c.v.data(), c.v.size() * 4, hipMemcpyHostToDevice));

  // useful KV bytes actually read: H * sum_s Tk(s) * hd * 2 * 4
  const double kv_bytes =
      double(c.H) * (double(c.S) * (c.T - c.S / 2.0 + 0.5)) * c.hd * 2 * 4;

  for (const char* scheme : {"v1", "A", "A2", "B"}) {
    auto run = [&] {
      if (c.hd == 64)
        launch_one<64>(scheme, c, dq, dk, dv, dout);
      else
        launch_one<128>(scheme, c, dq, dk, dv, dout);
    };
    run();  // correctness pass (also warms up)
    HIP_CHECK(hipDeviceSynchronize());
    if (write_out) {
      std::vector<float> host(c.q.size());
      HIP_CHECK(hipMemcpy(host.data(), dout, host.size() * 4, hipMemcpyDeviceToHost));
      std::string path = std::string(argv[1]) + ".out_" + scheme;
      FILE* f = std::fopen(path.c_str(), "wb");
      std::fwrite(host.data(), 4, host.size(), f);
      std::fclose(f);
    }
    // timing: adaptive iteration count targeting ~0.3 s per scheme
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0));
    run();
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float est_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&est_ms, ev0, ev1));
    const int iters = est_ms <= 0.0f ? 3 : std::max(3, std::min(200, int(300.0f / est_ms)));
    HIP_CHECK(hipEventRecord(ev0));
    for (int i = 0; i < iters; ++i) run();
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev0, ev1));
    const double ms = total_ms / iters;
    std::printf("hd=%-4uT=%-5uS=%-5uH=%-3u %-3s %8.3f ms  %7.1f GB/s KV\n", c.hd, c.T, c.S,
                c.H, scheme, ms, kv_bytes / ms / 1e6);
    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
  }
  return 0;
}
