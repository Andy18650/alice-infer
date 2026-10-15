// mixed_clock.hip — MFMA issue-rate test with independent interleaved VRAM
// reads: does memory traffic (rather than operand values) explain the real
// GEMM's lower effective clock?
//
// Context: the raw four-chain v_mfma_f32_32x32x8bf16 reference sustains
// ~1573 MHz effective with constant operands; the real CK V3 GEMM runs at
// ~1385 MHz at essentially the same flops/core-cycle (~250). The GEMM's
// requested global-load traffic is ~1.13 TB/s (64 KiB per CTA K-step per
// 8.39 MFLOP at 144 TFLOP/s). This experiment adds dummy VRAM reads to the
// reference loop — the loads are consumed only by a checksum, never by the
// MFMA operands — and measures how the effective clock responds.
//
// Kernel structure per loop iteration (per wave):
//   - consume the load issued kPipelineDepth-1 iterations earlier (xor into
//     a checksum), rotate the pipeline, issue one new coalesced load;
//   - four independent MFMA chains on register-constant operands.
// The depth-4 pipeline gives each load ~4 iterations (>1000 core cycles)
// before it is needed, so the MFMA stream never waits on memory unless the
// memory system itself saturates and latency grows beyond that.
//
// Load widths sweep the traffic axis (per thread per sub-step, per launch
// of 104 CTAs x 256 threads, kLoops=4096 sub-steps, ~0.7 ms if MFMA-bound):
//   0 B  -> 0 TB/s     (pure reference, control)
//   4 B  -> ~0.58 TB/s
//   8 B  -> ~1.17 TB/s (matches the real GEMM's requested traffic)
//   16 B -> ~2.3 TB/s  (above the ~1.29 TB/s measured read ceiling: the
//                       loop must become bus-limited)
// A loads-only control (no MFMA) measures the streaming ceiling of this
// harness for calibration.
//
// Loads walk a 512 MiB pseudo-random buffer (far above L2) in contiguous
// grid-wide blocks, mirroring the decode-roofline reader pattern, so the
// traffic is HBM streaming, not cache hits.
//
// Success criteria (user direction): keep flops/core-cycle >= 250 (matrix
// cores fully loaded) while loading the bus, and see whether the effective
// clock falls toward the real GEMM's ~1385 MHz. Negative results are
// reported as measured.
//
// Protocol: as the earlier clock studies — native -O3, one gfx90a GCD,
// automatic clocks, no clock locking, ~2 s warm-up per phase, 9 rounds of
// 16 event-timed launches, per-wave clock64()/wall_clock64() stamps, phase
// list forward and reversed, rocm_smi firmware-metrics monitor correlated
// via PHASE markers. MFMA output is verified exactly (128.0f for constant
// operands); the checksum must be deterministic across launches and
// non-constant across threads (guards against the loads being eliminated).

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
// Kernel parameters
// ---------------------------------------------------------------------------

constexpr int kThreads = 256;   // four wave64 warps per CTA
constexpr int kBlocks = 104;    // one CTA per CU on a 104-CU GCD
constexpr int kChains = 4;      // interleaved loop-carried MFMA chains
constexpr int kLoops = 4096;    // loop iterations (one MFMA per chain each)
constexpr int kRounds = 9;
constexpr int kTimedLaunches = 16;
constexpr int kWarmupSeconds = 2;
constexpr int kPipelineDepth = 8;  // load sub-steps before consumption

constexpr size_t kTotalThreads = size_t(kBlocks) * kThreads;
constexpr size_t kBufferBytes = size_t(512) << 20;  // 512 MiB, >> L2

// Raw counter readings bracketing one wave's kernel lifetime.
struct ClockStamp {
  uint64_t core_begin, steady_begin, core_end, steady_end;
};

// Cheap per-element hash (finalizer from the parent studies).
__device__ uint32_t hash_mix(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  return v ^ (v >> 16);
}

// Fill the read buffer with pseudo-random data (one-time setup).
__global__ void init_buffer(uint32_t* buffer, size_t words) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += size_t(gridDim.x) * blockDim.x)
    buffer[i] = hash_mix(uint32_t(i) ^ hash_mix(uint32_t(i >> 32)) ^ 0x9e37u);
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

using Bits4 = uint16_t __attribute__((ext_vector_type(4)));
using Float16v = float __attribute__((ext_vector_type(16)));
using U2 = uint32_t __attribute__((ext_vector_type(2)));
using U4 = uint32_t __attribute__((ext_vector_type(4)));

// LoadBytes: bytes read per thread per iteration (0, 4, 8 or 16).
// RandomOperands: per-lane pseudo-random BF16 operands instead of constant
// 2^-8 (the operand-value regime from the earlier studies).
// DoMfma: false = loads-only control (streaming ceiling calibration).
template <int LoadBytes, bool RandomOperands, bool DoMfma>
__global__ __launch_bounds__(kThreads) void mixed_clock_kernel(
    float* mfma_out, uint32_t* check_out, ClockStamp* stamps,
    const uint32_t* buffer) {
  static_assert(LoadBytes == 0 || LoadBytes == 4 || LoadBytes == 8 ||
                LoadBytes == 16);
  const uint64_t core_begin = clock64();
  const uint64_t steady_begin = wall_clock64();

  const uint32_t lane = threadIdx.x % 64;
  const size_t tid = size_t(blockIdx.x) * kThreads + threadIdx.x;

  // --- MFMA operands: constant 2^-8 BF16 or per-lane random values. ---
  Bits4 bits{};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (RandomOperands) {
      const uint32_t v = hash_mix((lane * 4 + j) ^ hash_mix(lane * 4 + j));
      bits[j] = 0x3e80u | (v & 127u) | ((v >> 16) & 0x8000u);
    } else {
      bits[j] = 0x3b80u;
    }
  }

  // --- MFMA accumulator chains (distinct initials for exact checking). ---
  Float16v c0{}, c1{}, c2{}, c3{};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    c1[j] = 1;
    c2[j] = 2;
    c3[j] = 3;
  }

  // --- Load pipeline: the loop below is manually unrolled into
  // kPipelineDepth sub-steps, each with its own named slot. Sub-step d
  // consumes slot[d]'s current value (issued exactly kPipelineDepth
  // sub-steps earlier, >2000 core cycles of cover) and immediately issues
  // the next load into the same slot. Constant slot indices (no rotation)
  // let the compiler emit precise partial waits instead of a full drain.
  // Addressing walks the whole buffer in contiguous grid-wide blocks (HBM
  // streaming, no cache reuse); consecutive threads read consecutive
  // elements.
  using Vec = std::conditional_t<LoadBytes == 16, U4,
                                 std::conditional_t<LoadBytes == 8, U2,
                                                    uint32_t>>;
  constexpr size_t kElements = kBufferBytes / (LoadBytes == 0 ? 1 : LoadBytes);
  constexpr size_t kMask = kElements - 1;  // kElements is a power of two
  const Vec* stream = reinterpret_cast<const Vec*>(buffer);
  auto load = [&](size_t progress) -> Vec {
    return stream[(progress * kTotalThreads + tid) & kMask];
  };
  auto fold = [](const Vec& v) -> uint32_t {
    if constexpr (LoadBytes == 16) return v[0] ^ v[1] ^ v[2] ^ v[3];
    else if constexpr (LoadBytes == 8) return v[0] ^ v[1];
    else return v;
  };

  Vec slot[kPipelineDepth]{};
  if (LoadBytes > 0) {
#pragma unroll
    for (int d = 0; d < kPipelineDepth; ++d) slot[d] = load(d);
  }
  uint32_t checksum = 0;

  // --- The hot loop: kPipelineDepth sub-steps per iteration, each sub-step
  // consuming one slot, issuing its replacement load, and running one MFMA
  // chain group on register-constant operands. The loaded data never
  // touches the MFMA operands. The outer loop is not unrolled so the
  // measured rate includes the real backedge; the inner loop is fully
  // unrolled (it is the software pipeline). ---
#pragma clang loop unroll(disable)
  for (int i = 0; i < kLoops; i += kPipelineDepth) {
#pragma unroll
    for (int d = 0; d < kPipelineDepth; ++d) {
      if (LoadBytes > 0) {
        checksum ^= fold(slot[d]);
        slot[d] = load(i + kPipelineDepth + d);
      }
      if (DoMfma) {
        c0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c3, 0, 0, 0);
      }
    }
  }

  // Drain the outstanding loads so none are eliminated.
  if (LoadBytes > 0) {
#pragma unroll
    for (int d = 0; d < kPipelineDepth; ++d) checksum ^= fold(slot[d]);
  }

  // --- Outputs: the MFMA reduction and the load checksum are stored
  // separately so both stay live and both can be verified. ---
  float mfma_sum = 0.0f;
  if (DoMfma) {
    const Float16v total = c0 + c1 + c2 + c3;
#pragma unroll
    for (int j = 0; j < 16; ++j) mfma_sum += total[j];
  }
  mfma_out[tid] = mfma_sum;
  check_out[tid] = checksum;

  // Per-wave timestamps, recorded after the hot loop.
  const uint64_t steady_end = wall_clock64();
  const uint64_t core_end = clock64();
  if (threadIdx.x % 64 == 0)
    stamps[size_t(blockIdx.x) * 4 + threadIdx.x / 64] = {
        core_begin, steady_begin, core_end, steady_end};
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct PhaseSpec {
  int load_bytes;
  bool random_operands;
  bool do_mfma;
  const char* name;
};

template <int LoadBytes>
void dispatch_modes(const PhaseSpec& spec, float* mfma_out,
                    uint32_t* check_out, ClockStamp* stamps,
                    const uint32_t* buffer, hipStream_t stream) {
  const dim3 grid(kBlocks), block(kThreads);
  const auto go = [&](auto random, auto mfma) {
    mixed_clock_kernel<LoadBytes, decltype(random)::value,
                       decltype(mfma)::value>
        <<<grid, block, 0, stream>>>(mfma_out, check_out, stamps, buffer);
  };
  if (spec.do_mfma) {
    if (spec.random_operands)
      go(std::true_type{}, std::true_type{});
    else
      go(std::false_type{}, std::true_type{});
  } else {
    if (spec.random_operands)
      go(std::true_type{}, std::false_type{});
    else
      go(std::false_type{}, std::false_type{});
  }
}

void launch_one(const PhaseSpec& spec, float* mfma_out, uint32_t* check_out,
                ClockStamp* stamps, const uint32_t* buffer,
                hipStream_t stream) {
  switch (spec.load_bytes) {
    case 0:
      dispatch_modes<0>(spec, mfma_out, check_out, stamps, buffer, stream);
      break;
    case 4:
      dispatch_modes<4>(spec, mfma_out, check_out, stamps, buffer, stream);
      break;
    case 8:
      dispatch_modes<8>(spec, mfma_out, check_out, stamps, buffer, stream);
      break;
    case 16:
      dispatch_modes<16>(spec, mfma_out, check_out, stamps, buffer, stream);
      break;
    default:
      throw std::runtime_error("bad load width");
  }
}

// Exact expected MFMA output: 128.0f for constant operands (see the earlier
// studies: 4096 iterations x 8 x (2^-8)^2 per lane, chains 0.5/1.5/2.5/3.5,
// 16 lanes), 0.0f for the loads-only control. Random operands are only
// checked for finiteness.
float expected_mfma(const PhaseSpec& spec) {
  if (!spec.do_mfma) return 0.0f;
  if (spec.random_operands) return -1.0f;  // finite check only
  return 128.0f;
}

void verify(const std::vector<float>& mfma, const std::vector<uint32_t>& check,
            const PhaseSpec& spec, const std::vector<float>* mfma_ref,
            const std::vector<uint32_t>* check_ref) {
  const float expected = expected_mfma(spec);
  if (expected >= 0.0f) {
    for (float v : mfma)
      if (v != expected)
        throw std::runtime_error("mixed-clock MFMA mismatch: got " +
                                 std::to_string(v) + ", expected " +
                                 std::to_string(expected));
  } else {
    // Random operands are lane-deterministic: outputs must repeat exactly.
    if (mfma_ref == nullptr)
      throw std::runtime_error("missing MFMA reference snapshot");
    if (mfma != *mfma_ref)
      throw std::runtime_error("random-operand MFMA determinism failure");
    for (float v : mfma)
      if (!std::isfinite(v))
        throw std::runtime_error("non-finite MFMA output");
  }
  if (spec.load_bytes > 0) {
    // The checksum must be deterministic across launches and must actually
    // depend on the loaded data (not all zero, varying across threads).
    if (check_ref != nullptr && check != *check_ref)
      throw std::runtime_error("checksum determinism failure");
    size_t nonzero = 0;
    for (uint32_t v : check) nonzero += v != 0;
    if (nonzero < check.size() / 2)
      throw std::runtime_error("checksum suspiciously constant — loads "
                               "possibly eliminated");
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct PhaseResult {
  double median_tops;
  double median_effective_mhz;
  double median_core_cycles;
  double flops_per_core_cycle;
  double median_gpu_us;
  double median_read_gbps;
};

PhaseResult run_phase(const PhaseSpec& spec, int pass, float* mfma_out,
                      uint32_t* check_out, ClockStamp* stamps,
                      const uint32_t* buffer, hipStream_t stream,
                      hipEvent_t start, hipEvent_t stop,
                      std::vector<float>& host_mfma,
                      std::vector<uint32_t>& host_check,
                      std::vector<ClockStamp>& host_stamps, int steady_khz) {
  const auto launch = [&] {
    launch_one(spec, mfma_out, check_out, stamps, buffer, stream);
  };
  const double flops =
      spec.do_mfma ? double(kBlocks) * (kThreads / 64) * kChains * kLoops *
                         16384.0
                   : 0.0;
  // Exact bytes read per launch: one load per thread per iteration plus the
  // pipeline prologue/drain.
  const double bytes = spec.load_bytes > 0
                           ? double(kLoops + kPipelineDepth) *
                                 kTotalThreads * spec.load_bytes
                           : 0.0;

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
    CHECK_HIP(hipStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - warmup_begin <
           std::chrono::seconds(kWarmupSeconds));

  // Correctness gate; snapshot references for determinism checks.
  launch();
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipMemcpy(host_mfma.data(), mfma_out,
                      host_mfma.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(host_check.data(), check_out,
                      host_check.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  const std::vector<float> mfma_reference = host_mfma;
  const std::vector<uint32_t> check_reference = host_check;
  verify(host_mfma, host_check, spec, &mfma_reference, &check_reference);

  // Timed rounds.
  std::vector<double> round_tops, round_gpu_us, round_gbps, wave_mhz,
      wave_cycles;
  for (int round = 0; round < kRounds; ++round) {
    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < kTimedLaunches; ++i) launch();
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float elapsed_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    const double gpu_us = double(elapsed_ms) * 1000.0 / kTimedLaunches;
    round_gpu_us.push_back(gpu_us);
    round_tops.push_back(flops / gpu_us / 1e6);
    round_gbps.push_back(bytes / (gpu_us * 1e3));

    CHECK_HIP(hipMemcpy(host_stamps.data(), stamps,
                        host_stamps.size() * sizeof(ClockStamp),
                        hipMemcpyDeviceToHost));
    for (const ClockStamp& s : host_stamps) {
      const double core = double(s.core_end - s.core_begin);
      const double steady = double(s.steady_end - s.steady_begin);
      if (steady <= 0) continue;
      wave_mhz.push_back(core / steady * steady_khz / 1000.0);
      wave_cycles.push_back(core);
    }
    CHECK_HIP(hipMemcpy(host_mfma.data(), mfma_out,
                        host_mfma.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(host_check.data(), check_out,
                        host_check.size() * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    verify(host_mfma, host_check, spec, &mfma_reference, &check_reference);
  }

  const double unix_end = std::chrono::duration<double>(
                               std::chrono::system_clock::now()
                                   .time_since_epoch())
                               .count();
  std::printf("PHASE_END pass=%d name=%s unix_s=%.3f\n", pass, spec.name,
              unix_end);
  std::fflush(stdout);

  PhaseResult result;
  result.median_tops = median(round_tops);
  result.median_effective_mhz = median(wave_mhz);
  result.median_core_cycles = median(wave_cycles);
  result.flops_per_core_cycle =
      spec.do_mfma
          ? double(kChains) * kLoops * 16384.0 / result.median_core_cycles
          : 0.0;
  result.median_gpu_us = median(round_gpu_us);
  result.median_read_gbps = median(round_gbps);
  return result;
}

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
  std::printf("protocol: automatic clocks, no locking, %d CTAs x %d threads, "
              "%d chains x %d loops, load pipeline depth %d, buffer %zu MiB\n",
              kBlocks, kThreads, kChains, kLoops, kPipelineDepth,
              kBufferBytes >> 20);
  std::fflush(stdout);

  hipStream_t stream;
  hipEvent_t start, stop;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  std::vector<float> host_mfma(kTotalThreads);
  std::vector<uint32_t> host_check(kTotalThreads);
  std::vector<ClockStamp> host_stamps(size_t(kBlocks) * 4);
  float* mfma_out = nullptr;
  uint32_t* check_out = nullptr;
  ClockStamp* stamps = nullptr;
  uint32_t* buffer = nullptr;
  CHECK_HIP(hipMalloc(&mfma_out, host_mfma.size() * sizeof(float)));
  CHECK_HIP(hipMalloc(&check_out, host_check.size() * sizeof(uint32_t)));
  CHECK_HIP(hipMalloc(&stamps, host_stamps.size() * sizeof(ClockStamp)));
  CHECK_HIP(hipMalloc(&buffer, kBufferBytes));
  init_buffer<<<832, 256, 0, stream>>>(buffer, kBufferBytes / 4);
  CHECK_HIP(hipStreamSynchronize(stream));

  // --- Phase list, forward and reversed. ---
  const std::vector<PhaseSpec> phases = {
      {0, false, true, "bf16-load0B"},
      {4, false, true, "bf16-load4B"},
      {8, false, true, "bf16-load8B"},
      {16, false, true, "bf16-load16B"},
      {8, true, true, "bf16-random-load8B"},
      {16, false, false, "loadonly-16B"},
  };

  std::printf("\n%-8s %-22s %10s %10s %12s %10s %10s %10s\n", "pass", "phase",
              "TOPS", "eff_MHz", "core_cycles", "flop/cyc", "read_GBps",
              "gpu_us");
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t i = 0; i < phases.size(); ++i) {
      const PhaseSpec& spec = pass == 0 ? phases[i]
                                        : phases[phases.size() - 1 - i];
      const PhaseResult r =
          run_phase(spec, pass, mfma_out, check_out, stamps, buffer, stream,
                    start, stop, host_mfma, host_check, host_stamps,
                    steady_khz);
      std::printf("%-8d %-22s %10.3f %10.2f %12.0f %10.2f %10.1f %10.3f\n",
                  pass, spec.name, r.median_tops, r.median_effective_mhz,
                  r.median_core_cycles, r.flops_per_core_cycle,
                  r.median_read_gbps, r.median_gpu_us);
      std::fflush(stdout);
    }
  }

  CHECK_HIP(hipFree(buffer));
  CHECK_HIP(hipFree(stamps));
  CHECK_HIP(hipFree(check_out));
  CHECK_HIP(hipFree(mfma_out));
  CHECK_HIP(hipEventDestroy(stop));
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipStreamDestroy(stream));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return 1;
}
