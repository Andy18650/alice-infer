// operand_clock.hip — is the operand-dependent effective-clock effect
// repeatable, and does it appear for FP16 as well as BF16?
//
// The parent study (clock_audit.hip) found that the raw-MFMA reference runs
// at a higher effective clock with constant low-magnitude BF16 operands
// (~1571 MHz) than with normal signed BF16 operands (~1426 MHz), with
// throttle bits zero and automatic clocks. That is a suspicious result, so
// this experiment repeats it under controlled conditions:
//
//   - Same kernel structure for every configuration: the four-chain
//     v_mfma_f32_32x32x8 loop, no memory traffic in the hot loop, 104 CTAs.
//   - Two instruction flavors: the BF16 and the FP16 variant.
//   - Three operand regimes:
//       ConstantTiny    — every operand half is exactly 2^-8 (the parent
//                         study's constant; all products are powers of two)
//       ConstantQuarter — every operand half is exactly 0.25 (same magnitude
//                         as the random regime, but no sign flips; separates
//                         "magnitude" from "randomness")
//       Random          — per-lane pseudo-random signed values of magnitude
//                         ~0.25 (the parent study's random regime)
//   - Effective clock measured the same way as the parent study: each wave
//     brackets itself with clock64() (core cycles, frequency varies) and
//     wall_clock64() (fixed-rate steady counter); effective MHz is the
//     cycles-to-ticks ratio times the steady counter rate.
//   - The full phase list runs forward and then in reverse order, to check
//     that results do not depend on phase ordering.
//
// Protocol: native -O3, one gfx90a GCD, automatic clocks, no clock locking,
// CPU+memory NUMA node 0. ~2 s warm-up per phase, then 9 rounds of 16
// event-timed launches per phase. Constant phases are verified against exact
// expected outputs; random phases are checked for finiteness.

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
// Kernel parameters (identical structure to the parent study's reference)
// ---------------------------------------------------------------------------

constexpr int kThreads = 256;  // four wave64 warps per CTA
constexpr int kBlocks = 104;   // one CTA per CU on a 104-CU GCD
constexpr int kChains = 4;     // interleaved loop-carried accumulator chains
constexpr int kLoops = 4096;   // MFMA iterations per chain
constexpr int kRounds = 9;
constexpr int kTimedLaunches = 16;
constexpr int kWarmupSeconds = 2;

// ---------------------------------------------------------------------------
// Operand regimes
// ---------------------------------------------------------------------------

enum class OperandMode { ConstantTiny, ConstantQuarter, Random };

// Raw counter readings bracketing one wave's kernel lifetime.
struct ClockStamp {
  uint64_t core_begin, steady_begin, core_end, steady_end;
};

// Cheap per-lane hash (finalizer from the parent study) so random operands
// differ per lane with no memory traffic.
__device__ uint32_t hash_mix(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  return v ^ (v >> 16);
}

// Encoding of 2^-8 in each format: BF16 and FP16 both represent it exactly.
constexpr uint16_t kBf16Tiny = 0x3b80;
constexpr uint16_t kFp16Tiny = 0x1c00;
// Encoding of 0.25 in each format (also exact).
constexpr uint16_t kBf16Quarter = 0x3e80;
constexpr uint16_t kFp16Quarter = 0x3400;

// Bits for one operand half. Random mode mirrors the parent study: ~0.25
// magnitude with low-bit jitter and a random sign, so the values look like
// normal signed training-style data rather than special constants.
__device__ uint16_t operand_bits(bool is_fp16, OperandMode mode,
                                 uint32_t index) {
  switch (mode) {
    case OperandMode::ConstantTiny:
      return is_fp16 ? kFp16Tiny : kBf16Tiny;
    case OperandMode::ConstantQuarter:
      return is_fp16 ? kFp16Quarter : kBf16Quarter;
    case OperandMode::Random: {
      const uint32_t v = hash_mix(index ^ hash_mix(index >> 16));
      const uint16_t base = is_fp16 ? kFp16Quarter : kBf16Quarter;
      const uint16_t jitter_mask = is_fp16 ? 63 : 127;  // stay in mantissa
      return base | (v & jitter_mask) | ((v >> 16) & 0x8000u);
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// The kernel: four MFMA chains, nothing else in the hot loop
// ---------------------------------------------------------------------------

using Bits4 = uint16_t __attribute__((ext_vector_type(4)));
using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Float16v = float __attribute__((ext_vector_type(16)));

template <bool IsFp16, OperandMode Mode>
__global__ __launch_bounds__(kThreads) void mfma_operand_clock_kernel(
    float* out, ClockStamp* stamps) {
  const uint64_t core_begin = clock64();
  const uint64_t steady_begin = wall_clock64();

  // Operand prologue (outside the timed loop): per-lane values for Random
  // mode, compile-time constants otherwise.
  const uint32_t lane = threadIdx.x % 64;
  Bits4 bits{};
#pragma unroll
  for (int j = 0; j < 4; ++j)
    bits[j] = operand_bits(IsFp16, Mode, lane * 4 + j);

  // Distinct chain initials keep the chains distinguishable and give exact
  // expected outputs for the constant modes.
  Float16v c0{}, c1{}, c2{}, c3{};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    c1[j] = 1;
    c2[j] = 2;
    c3[j] = 3;
  }

  // The hot loop: MFMA instructions and loop control only. Unrolling is
  // disabled so the measured rate includes the real backedge.
#pragma clang loop unroll(disable)
  for (int i = 0; i < kLoops; ++i) {
    if constexpr (IsFp16) {
      const Half4 a = __builtin_bit_cast(Half4, bits);
      // Note: the FP16 flavor of the builtin has no _1k suffix.
      c0 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, c0, 0, 0, 0);
      c1 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, c1, 0, 0, 0);
      c2 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, c2, 0, 0, 0);
      c3 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, a, c3, 0, 0, 0);
    } else {
      c0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c0, 0, 0, 0);
      c1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c1, 0, 0, 0);
      c2 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c2, 0, 0, 0);
      c3 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bits, bits, c3, 0, 0, 0);
    }
  }

  // Reduce and store one float per thread (defeats dead-code elimination).
  const Float16v total = c0 + c1 + c2 + c3;
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) sum += total[j];
  out[size_t(blockIdx.x) * kThreads + threadIdx.x] = sum;

  // Per-wave timestamps, recorded after the hot loop so it is not perturbed.
  const uint64_t steady_end = wall_clock64();
  const uint64_t core_end = clock64();
  if (threadIdx.x % 64 == 0)
    stamps[size_t(blockIdx.x) * 4 + threadIdx.x / 64] = {
        core_begin, steady_begin, core_end, steady_end};
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Dispatch over the six kernel instantiations.
void launch_one(bool is_fp16, OperandMode mode, float* out,
                ClockStamp* stamps, hipStream_t stream) {
  const dim3 grid(kBlocks), block(kThreads);
  if (is_fp16) {
    switch (mode) {
      case OperandMode::ConstantTiny:
        mfma_operand_clock_kernel<true, OperandMode::ConstantTiny>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::ConstantQuarter:
        mfma_operand_clock_kernel<true, OperandMode::ConstantQuarter>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::Random:
        mfma_operand_clock_kernel<true, OperandMode::Random>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
    }
  } else {
    switch (mode) {
      case OperandMode::ConstantTiny:
        mfma_operand_clock_kernel<false, OperandMode::ConstantTiny>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::ConstantQuarter:
        mfma_operand_clock_kernel<false, OperandMode::ConstantQuarter>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::Random:
        mfma_operand_clock_kernel<false, OperandMode::Random>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
    }
  }
}

// Exact expected per-thread output for the constant modes. All products are
// powers of two and all sums stay far below 2^24, so FP32 arithmetic is exact
// and any mismatch means real breakage.
//   Tiny:    each instruction adds 8 * (2^-8)^2 = 2^-13 to every accumulator
//            lane; after 4096 iterations the chains hold 0.5/1.5/2.5/3.5,
//            their lane sum is 8, and 16 lanes give 128.
//   Quarter: each instruction adds 8 * 0.25^2 = 0.5; after 4096 iterations
//            the chains hold 2048/2049/2050/2051, the lane sum is 8198, and
//            16 lanes give 131168.
float expected_output(OperandMode mode) {
  if (mode == OperandMode::ConstantTiny) return 128.0f;
  if (mode == OperandMode::ConstantQuarter) return 131168.0f;
  return 0.0f;  // random mode: only finiteness is checked
}

void verify(const std::vector<float>& values, OperandMode mode) {
  if (mode == OperandMode::Random) {
    for (float v : values)
      if (!std::isfinite(v))
        throw std::runtime_error("non-finite output in random mode");
    return;
  }
  const float expected = expected_output(mode);
  for (float v : values)
    if (v != expected)
      throw std::runtime_error("operand-clock kernel mismatch: got " +
                               std::to_string(v) + ", expected " +
                               std::to_string(expected));
}

// FLOPs in one launch: CTAs * warps * chains * loops * (32*32*8*2).
double flops_per_launch() {
  return double(kBlocks) * (kThreads / 64) * kChains * kLoops * 16384.0;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct PhaseResult {
  double median_tflops;
  double median_effective_mhz;
  double median_core_cycles;
  double flops_per_core_cycle;
};

// One measurement phase: warm up, verify once, then collect timed rounds
// with per-wave clock stamps. Prints PHASE markers (with unix time) so an
// external monitor's samples can be attributed to phases afterwards.
PhaseResult run_phase(bool is_fp16, OperandMode mode, const char* name,
                      int pass, float* out, ClockStamp* stamps,
                      hipStream_t stream, hipEvent_t start, hipEvent_t stop,
                      std::vector<float>& host_out,
                      std::vector<ClockStamp>& host_stamps, int steady_khz) {
  const auto launch = [&] { launch_one(is_fp16, mode, out, stamps, stream); };
  const double flops = flops_per_launch();

  // Phase marker for external monitor correlation.
  const double unix_begin = std::chrono::duration<double>(
                                 std::chrono::system_clock::now()
                                     .time_since_epoch())
                                 .count();
  std::printf("PHASE_BEGIN pass=%d name=%s unix_s=%.3f\n", pass, name,
              unix_begin);
  std::fflush(stdout);

  // Warm-up: ~2 s of the exact same kernel, to settle clocks and caches.
  const auto warmup_begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 8; ++i) launch();
    CHECK_HIP(hipStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - warmup_begin <
           std::chrono::seconds(kWarmupSeconds));

  // Correctness gate before timing.
  launch();
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipMemcpy(host_out.data(), out, host_out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  verify(host_out, mode);

  // Timed rounds. After each round the stamp buffer holds the last launch's
  // per-wave readings.
  std::vector<double> round_tflops, wave_mhz, wave_cycles;
  for (int round = 0; round < kRounds; ++round) {
    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < kTimedLaunches; ++i) launch();
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float elapsed_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    round_tflops.push_back(flops / (double(elapsed_ms) * 1000.0 /
                                    kTimedLaunches) /
                           1e6);

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
  }

  const double unix_end = std::chrono::duration<double>(
                               std::chrono::system_clock::now()
                                   .time_since_epoch())
                               .count();
  std::printf("PHASE_END pass=%d name=%s unix_s=%.3f\n", pass, name,
              unix_end);
  std::fflush(stdout);

  PhaseResult result;
  result.median_tflops = median(round_tflops);
  result.median_effective_mhz = median(wave_mhz);
  result.median_core_cycles = median(wave_cycles);
  // Per-wave FLOPs / core cycles: instruction throughput per cycle, the
  // clock-independent utilization measure.
  result.flops_per_core_cycle =
      double(kChains) * kLoops * 16384.0 / result.median_core_cycles;
  return result;
}

}  // namespace

int main(int argc, char** argv) try {
  for (int i = 1; i < argc; ++i)
    throw std::runtime_error(std::string("unknown argument ") + argv[i]);

  // --- Device check: the MFMA builtins below are gfx90a-specific. ---
  CHECK_HIP(hipSetDevice(0));
  hipDeviceProp_t props{};
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  if (std::string(props.gcnArchName).find("gfx90a") == std::string::npos ||
      props.warpSize != 64)
    throw std::runtime_error("requires gfx90a wave64");

  // The fixed rate of wall_clock64(), needed to turn the cycles-to-ticks
  // ratio into a frequency.
  int steady_khz = 0;
  CHECK_HIP(hipDeviceGetAttribute(&steady_khz,
                                  hipDeviceAttributeWallClockRate, 0));

  std::printf("device=0 arch=%s CUs=%d steady_clock_khz=%d\n",
              props.gcnArchName, props.multiProcessorCount, steady_khz);
  std::printf("protocol: automatic clocks, no locking, %d CTAs x %d threads, "
              "%d chains x %d loops, %d rounds x %d launches\n",
              kBlocks, kThreads, kChains, kLoops, kRounds, kTimedLaunches);
  std::fflush(stdout);

  // --- Runtime objects. ---
  hipStream_t stream;
  hipEvent_t start, stop;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  std::vector<float> host_out(size_t(kBlocks) * kThreads);
  std::vector<ClockStamp> host_stamps(size_t(kBlocks) * 4);
  float* out = nullptr;
  ClockStamp* stamps = nullptr;
  CHECK_HIP(hipMalloc(&out, host_out.size() * sizeof(float)));
  CHECK_HIP(hipMalloc(&stamps, host_stamps.size() * sizeof(ClockStamp)));

  // --- Phase list: both dtypes x three operand regimes, forward and
  // reversed, so ordering dependence would show up as a mismatch. ---
  struct PhaseSpec {
    bool is_fp16;
    OperandMode mode;
    const char* name;
  };
  const std::vector<PhaseSpec> phases = {
      {false, OperandMode::ConstantTiny, "bf16-constant-tiny"},
      {false, OperandMode::ConstantQuarter, "bf16-constant-quarter"},
      {false, OperandMode::Random, "bf16-random"},
      {true, OperandMode::ConstantTiny, "fp16-constant-tiny"},
      {true, OperandMode::ConstantQuarter, "fp16-constant-quarter"},
      {true, OperandMode::Random, "fp16-random"},
  };

  std::printf("\n%-8s %-24s %12s %12s %12s %14s\n", "pass", "phase",
              "TFLOP/s", "eff_MHz", "core_cycles", "flop/cycle");
  for (int pass = 0; pass < 2; ++pass) {
    // Pass 0 runs the list forward, pass 1 in reverse.
    for (size_t i = 0; i < phases.size(); ++i) {
      const PhaseSpec& spec = pass == 0 ? phases[i]
                                        : phases[phases.size() - 1 - i];
      const PhaseResult r =
          run_phase(spec.is_fp16, spec.mode, spec.name, pass, out, stamps,
                    stream, start, stop, host_out, host_stamps, steady_khz);
      std::printf("%-8d %-24s %12.3f %12.2f %12.0f %14.2f\n", pass,
                  spec.name, r.median_tflops, r.median_effective_mhz,
                  r.median_core_cycles, r.flops_per_core_cycle);
      std::fflush(stdout);
    }
  }

  CHECK_HIP(hipFree(stamps));
  CHECK_HIP(hipFree(out));
  CHECK_HIP(hipEventDestroy(stop));
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipStreamDestroy(stream));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return 1;
}
