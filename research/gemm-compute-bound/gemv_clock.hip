// gemv_clock.hip — the optimized research GEMV (packed split-K, SIMT only)
// under the effective-clock measurement protocol.
//
// Kernel source: research/decode-roofline/gemv_cm_sk.hip ([10] in that
// study), copied nearly verbatim. It is the bandwidth-saturating research
// GEMV: one thread per (row, K-chunk) over a packed column-pair matrix
// (uint32 = 2 bf16), 4 FMA accumulators, no MFMA, no shuffle — plain SIMT
// loads and VALU math. Best measured rates: gup ~1277 GB/s, down ~1297
// GB/s (99% of the 1290 GB/s streaming ceiling).
//
// Additions (marked): per-wave clock64()/wall_clock64() stamps and the
// standard clock-study host protocol (2 s warm-up, 9 rounds x 16 event-
// timed launches, per-wave effective MHz, PHASE markers with unix
// timestamps, exact-oracle correctness, rocm_smi firmware monitor).
//
// Question: where does the effective clock sit for a real SIMT kernel
// running at HBM saturation with data-dependent math on every loaded byte?
// Reference points from the MFMA studies: loads-only 1227 GB/s -> 1614
// MHz; MFMA + 1.2 TB/s + constant operands -> 1510 MHz; MFMA + traffic +
// random operands -> 1388 MHz; real MFMA GEMM -> 1385 MHz.
//
// Phases: the four study shapes at their bandwidth-optimal chunk widths
// plus extra CW points for a bandwidth sweep, and a zero-weights gup
// phase (all weight bits zero; FMAs still execute) to separate DRAM data
// content from pure traffic, mirroring the operand-regime controls.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_HIP(call)                                                    \
  do {                                                                     \
    hipError_t status = (call);                                            \
    if (status != hipSuccess) {                                            \
      throw std::runtime_error(std::string(#call) + ": " +                 \
                               hipGetErrorString(status));                 \
    }                                                                      \
  } while (0)

namespace {

// ---------------------------------------------------------------------------
// Protocol parameters (as in the other clock studies)
// ---------------------------------------------------------------------------

constexpr int kRounds = 9;
constexpr int kTimedLaunches = 16;
constexpr int kWarmupSeconds = 2;
constexpr uint32_t kThreadsPerCta = 64;  // native wave, one wave per CTA
constexpr uint32_t kMaxCtas = 16384;     // stamp buffer bound

// Raw counter readings bracketing one wave's kernel lifetime.
struct ClockStamp {
  uint64_t core_begin, steady_begin, core_end, steady_end;
};

// ---------------------------------------------------------------------------
// The GEMV kernel — copied from research/decode-roofline/gemv_cm_sk.hip.
// Only the ClockStamp additions are new (marked).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float b2f(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ float qbf16(float f) {
  return __uint_as_float(__float_as_uint(f) & 0xffff0000u);
}

// one thread per (row, K-chunk); uint32 (2 bf16) per step from the packed
// matrix; partials[c*N + n] = row n's sum over chunk c. No shuffle.
__global__ void k_gemv_cm_sk(const uint32_t* __restrict__ Wp,
                             const float* __restrict__ x,
                             float* __restrict__ partials,
                             ClockStamp* __restrict__ stamps,  // added
                             uint32_t N, uint32_t K, uint32_t CW) {
  const uint64_t core_begin = clock64();             // added
  const uint64_t steady_begin = wall_clock64();      // added
  const uint32_t chunks = K / CW;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * chunks) return;
  const uint32_t n = idx % N;
  const uint32_t c = idx / N;
  const uint32_t p0 = c * (CW >> 1), p1 = p0 + (CW >> 1);
  const float4* xr = reinterpret_cast<const float4*>(x);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (uint32_t p = p0; p < p1; p += 4) {
    const uint32_t w0 = Wp[(size_t)p * N + n];
    const uint32_t w1 = Wp[(size_t)(p + 1) * N + n];
    const uint32_t w2 = Wp[(size_t)(p + 2) * N + n];
    const uint32_t w3 = Wp[(size_t)(p + 3) * N + n];
    const float4 xa = xr[p >> 1], xb = xr[(p >> 1) + 1];
    a0 += b2f((unsigned short)(w0 & 0xffff)) * qbf16(xa.x) +
          b2f((unsigned short)(w0 >> 16)) * qbf16(xa.y);
    a1 += b2f((unsigned short)(w1 & 0xffff)) * qbf16(xa.z) +
          b2f((unsigned short)(w1 >> 16)) * qbf16(xa.w);
    a2 += b2f((unsigned short)(w2 & 0xffff)) * qbf16(xb.x) +
          b2f((unsigned short)(w2 >> 16)) * qbf16(xb.y);
    a3 += b2f((unsigned short)(w3 & 0xffff)) * qbf16(xb.z) +
          b2f((unsigned short)(w3 >> 16)) * qbf16(xb.w);
  }
  partials[(size_t)c * N + n] = a0 + a1 + a2 + a3;
  const uint64_t steady_end = wall_clock64();     // added
  const uint64_t core_end = clock64();            // added
  if (threadIdx.x == 0)  // added: one stamp per CTA (single wave)
    stamps[blockIdx.x] = {core_begin, steady_begin, core_end, steady_end};
}

// correctness only: y[n] = sum_c partials[c*N + n]  (verbatim)
__global__ void k_combine(const float* __restrict__ partials,
                          float* __restrict__ y, uint32_t N, uint32_t chunks) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.f;
  for (uint32_t c = 0; c < chunks; ++c) a += partials[(size_t)c * N + n];
  y[n] = a;
}

// ---------------------------------------------------------------------------
// Host side: shapes, packing, standard measurement protocol
// ---------------------------------------------------------------------------

struct ShapeSpec {
  const char* name;
  uint32_t N, K, CW;
  bool zero_weights;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct PhaseResult {
  double median_gbps;
  double median_tflops;
  double median_effective_mhz;
  double median_core_cycles;
  double median_gpu_us;
};

class GemvClock {
 public:
  GemvClock(uint32_t max_n, uint32_t max_k)
      : max_n_(max_n), max_k_(max_k) {
    CHECK_HIP(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    CHECK_HIP(hipEventCreate(&start_));
    CHECK_HIP(hipEventCreate(&stop_));
    // Host-side activation vector, as in the study (k % 7 pattern).
    std::vector<float> hx(max_k);
    for (uint32_t i = 0; i < max_k; ++i) hx[i] = float(i % 7);
    CHECK_HIP(hipMalloc(&packed_, size_t(max_n) * (max_k / 2) * 4));
    CHECK_HIP(hipMalloc(&x_, hx.size() * 4));
    CHECK_HIP(hipMalloc(&partials_, size_t(64) * max_n * 4));
    CHECK_HIP(hipMalloc(&y_, size_t(max_n) * 4));
    CHECK_HIP(hipMalloc(&stamps_, size_t(kMaxCtas) * sizeof(ClockStamp)));
    CHECK_HIP(hipMemcpy(x_, hx.data(), hx.size() * 4,
                        hipMemcpyHostToDevice));
  }

  ~GemvClock() {
    (void)hipFree(stamps_);
    (void)hipFree(y_);
    (void)hipFree(partials_);
    (void)hipFree(x_);
    (void)hipFree(packed_);
    (void)hipEventDestroy(stop_);
    (void)hipEventDestroy(start_);
    (void)hipStreamDestroy(stream_);
  }

  // Pack the matrix for one phase: the study's (2p)%13 pattern, or all
  // zero bits for the zero-weights control. Outside timing.
  void pack(const ShapeSpec& spec) {
    std::vector<uint32_t> hwp((size_t)spec.N * (spec.K / 2), 0);
    if (!spec.zero_weights) {
      for (uint32_t p = 0; p < spec.K / 2; ++p) {
        float f0 = float((2 * p) % 13), f1 = float((2 * p + 1) % 13);
        uint32_t lo, hi, bits;
        memcpy(&bits, &f0, 4);
        lo = bits >> 16;
        memcpy(&bits, &f1, 4);
        hi = bits >> 16;
        for (uint32_t n = 0; n < spec.N; ++n)
          hwp[(size_t)p * spec.N + n] = lo | (hi << 16);
      }
    }
    CHECK_HIP(hipMemcpyAsync(packed_, hwp.data(), hwp.size() * 4,
                             hipMemcpyHostToDevice, stream_));
    CHECK_HIP(hipStreamSynchronize(stream_));
  }

  // One measured phase: warm up, verify, then timed rounds with stamps.
  PhaseResult run(const ShapeSpec& spec, int pass, int steady_khz) {
    const uint32_t chunks = spec.K / spec.CW;
    const uint32_t threads = spec.N * chunks;
    const uint32_t grid = (threads + kThreadsPerCta - 1) / kThreadsPerCta;
    if (grid > kMaxCtas)
      throw std::runtime_error("stamp buffer too small for grid");
    const auto launch = [&] {
      k_gemv_cm_sk<<<grid, kThreadsPerCta, 0, stream_>>>(
          packed_, x_, partials_, stamps_, spec.N, spec.K, spec.CW);
    };
    // Useful weight bytes read per launch (the study's bandwidth metric);
    // partials writes and x reads are ~1%.
    const double weight_bytes = double(spec.N) * spec.K * 2;
    const double tflops = 2.0 * spec.N * spec.K / 1e6;  // MFLOP per launch;
    // MFLOP / us == TFLOP/s.

    pack(spec);

    const double unix_begin = std::chrono::duration<double>(
                                  std::chrono::system_clock::now()
                                      .time_since_epoch())
                                  .count();
    std::printf("PHASE_BEGIN pass=%d name=%s unix_s=%.3f\n", pass, spec.name,
                unix_begin);
    std::fflush(stdout);

    // Warm-up: ~2 s of the exact same kernel.
    const auto warmup_begin = std::chrono::steady_clock::now();
    do {
      for (int i = 0; i < 8; ++i) launch();
      CHECK_HIP(hipStreamSynchronize(stream_));
    } while (std::chrono::steady_clock::now() - warmup_begin <
             std::chrono::seconds(kWarmupSeconds));

    // Correctness gate (study's oracle: combine row 0 and compare).
    verify(spec);

    // Timed rounds.
    std::vector<double> round_gbps, round_tflops, round_gpu_us, wave_mhz,
        wave_cycles;
    std::vector<ClockStamp> host_stamps(grid);
    for (int round = 0; round < kRounds; ++round) {
      CHECK_HIP(hipEventRecord(start_, stream_));
      for (int i = 0; i < kTimedLaunches; ++i) launch();
      CHECK_HIP(hipEventRecord(stop_, stream_));
      CHECK_HIP(hipEventSynchronize(stop_));
      float ms = 0;
      CHECK_HIP(hipEventElapsedTime(&ms, start_, stop_));
      const double gpu_us = double(ms) * 1000.0 / kTimedLaunches;
      round_gpu_us.push_back(gpu_us);
      round_gbps.push_back(weight_bytes / (gpu_us * 1e3));
      round_tflops.push_back(tflops / gpu_us);

      CHECK_HIP(hipMemcpyAsync(host_stamps.data(), stamps_,
                               grid * sizeof(ClockStamp),
                               hipMemcpyDeviceToHost, stream_));
      CHECK_HIP(hipStreamSynchronize(stream_));
      for (const ClockStamp& s : host_stamps) {
        const double core = double(s.core_end - s.core_begin);
        const double steady = double(s.steady_end - s.steady_begin);
        if (steady <= 0) continue;
        wave_mhz.push_back(core / steady * steady_khz / 1000.0);
        wave_cycles.push_back(core);
      }
    }

    const double unix_end = std::chrono::duration<double>(
                                std::chrono::system_clock::now()
                                    .time_since_epoch())
                                .count();
    std::printf("PHASE_END pass=%d name=%s unix_s=%.3f\n", pass, spec.name,
                unix_end);
    std::fflush(stdout);

    PhaseResult r;
    r.median_gbps = median(round_gbps);
    r.median_tflops = median(round_tflops);
    r.median_effective_mhz = median(wave_mhz);
    r.median_core_cycles = median(wave_cycles);
    r.median_gpu_us = median(round_gpu_us);
    return r;
  }

 private:
  // Runs one GEMV + combine and checks y[0] against the study's oracle
  // (exact zeros for the zero-weights control).
  void verify(const ShapeSpec& spec) {
    const uint32_t chunks = spec.K / spec.CW;
    const uint32_t threads = spec.N * chunks;
    const uint32_t grid = (threads + kThreadsPerCta - 1) / kThreadsPerCta;
    k_gemv_cm_sk<<<grid, kThreadsPerCta, 0, stream_>>>(
        packed_, x_, partials_, stamps_, spec.N, spec.K, spec.CW);
    k_combine<<<(spec.N + 255) / 256, 256, 0, stream_>>>(partials_, y_,
                                                         spec.N, chunks);
    CHECK_HIP(hipStreamSynchronize(stream_));
    float y0 = 0;
    CHECK_HIP(hipMemcpy(&y0, y_, 4, hipMemcpyDeviceToHost));
    double expect = 0;
    if (!spec.zero_weights) {
      for (uint32_t k = 0; k < spec.K; ++k) {
        float xk = float(k % 7);
        uint32_t bits;
        memcpy(&bits, &xk, 4);
        bits &= 0xffff0000u;
        float xq;
        memcpy(&xq, &bits, 4);
        expect += double(k % 13) * double(xq);
      }
    }
    const double err = std::fabs(double(y0) - expect);
    if (!(err < 1.0))
      throw std::runtime_error("gemv oracle mismatch: y[0]=" +
                               std::to_string(y0) + " expected " +
                               std::to_string(expect));
  }

  uint32_t max_n_, max_k_;
  hipStream_t stream_{};
  hipEvent_t start_{}, stop_{};
  uint32_t* packed_ = nullptr;
  float* x_ = nullptr;
  float* partials_ = nullptr;
  float* y_ = nullptr;
  ClockStamp* stamps_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) try {
  for (int i = 1; i < argc; ++i)
    throw std::runtime_error(std::string("unknown argument ") + argv[i]);

  CHECK_HIP(hipSetDevice(0));
  hipDeviceProp_t props{};
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  if (std::string(props.gcnArchName).find("gfx90a") == std::string::npos ||
      props.warpSize != 64)
    throw std::runtime_error("requires gfx90a wave64");
  int steady_khz = 0;
  CHECK_HIP(hipDeviceGetAttribute(&steady_khz,
                                  hipDeviceAttributeWallClockRate, 0));
  std::printf("device=0 arch=%s CUs=%d steady_clock_khz=%d\n",
              props.gcnArchName, props.multiProcessorCount, steady_khz);
  std::printf("protocol: automatic clocks, no locking, %d rounds x %d "
              "launches, ~%d s warm-up per phase\n",
              kRounds, kTimedLaunches, kWarmupSeconds);
  std::fflush(stdout);

  // Shapes and chunk widths from the decode-roofline [10] study; CW sweep
  // points chosen to spread the bandwidth axis (~490 to ~1300 GB/s).
  const std::vector<ShapeSpec> phases = {
      {"gup-cw256", 14336, 8192, 256, false},
      {"gup-cw512", 14336, 8192, 512, false},
      {"down-cw512", 8192, 7168, 512, false},
      {"down-cw256", 8192, 7168, 256, false},
      {"qkv-cw256", 2560, 8192, 256, false},
      {"o-cw256", 2048, 8192, 256, false},
      {"o-cw1024", 2048, 8192, 1024, false},
      {"gup-cw256-zero", 14336, 8192, 256, true},
  };

  GemvClock bench(14336, 8192);
  std::printf("\n%-8s %-16s %10s %10s %12s %10s %10s\n", "pass", "phase",
              "GB/s", "TFLOP/s", "eff_MHz", "core_cyc", "gpu_us");
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t i = 0; i < phases.size(); ++i) {
      const ShapeSpec& spec = pass == 0 ? phases[i]
                                        : phases[phases.size() - 1 - i];
      const PhaseResult r = bench.run(spec, pass, steady_khz);
      std::printf("%-8d %-16s %10.1f %10.2f %12.2f %10.0f %10.3f\n", pass,
                  spec.name, r.median_gbps, r.median_tflops,
                  r.median_effective_mhz, r.median_core_cycles,
                  r.median_gpu_us);
      std::fflush(stdout);
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return 1;
}
