// int8_clock.hip — the operand-clock experiment for INT8 MFMA instructions.
//
// Follow-up to operand_clock.hip (BF16/FP16): the same four-chain MFMA
// loop, the same per-wave dual-counter clock measurement, and the same
// operand-regime controls, but with the gfx90a INT8 matrix instructions:
//
//   v_mfma_i32_32x32x8i8  — 16 i32 accumulator lanes per thread, 8192 MACs
//                            per instruction: the direct structural match of
//                            the BF16/FP16 v_mfma_f32_32x32x8 test (same
//                            output tile, same MACs per issue, but half the
//                            operand register bytes: 4 int8s in one 32-bit
//                            register instead of 4 halves in 64 bits).
//   v_mfma_i32_16x16x16i8 — 4 accumulator lanes, 4096 MACs per instruction
//                            (matches the 16x16x16f16 raw-issue flavor).
//
// Hypothesis under test (user direction): INT8 moves less register data and
// uses smaller multipliers, so it should draw less power and may therefore
// sustain a higher clock (or better throughput) than the float flavors.
// Negative results are reported as measured.
//
// Note on instruction selection: with this toolchain (clang 21, gfx90a) the
// CDNA1-style double-K INT8 flavors (v_mfma_i32_32x32x16i8, 16x16x32i8)
// fail instruction selection; the two instructions above are the ones that
// compile and whose disassembly matches their names (verified via
// -save-temps: 4 resp. 16 accumulator registers per instruction).
//
// Operand regimes (INT8 analogues of the float study's regimes):
//   ConstantOne — every operand byte is 1 (minimal bit toggling)
//   ConstantMid — every operand byte is 63 (mid-range constant)
//   Random      — per-lane pseudo-random signed bytes over the full range
//
// Verification: the i32 accumulators wrap modulo 2^32 (two's complement,
// as the hardware does). Constant modes are checked exactly against a
// modular expected value; the random mode is checked for run-to-run
// determinism (operands depend only on the lane index).
//
// Protocol: identical to operand_clock.hip — native -O3, one gfx90a GCD,
// automatic clocks, no clock locking, ~2 s warm-up per phase, 9 rounds of
// 16 event-timed launches, per-wave clock64()/wall_clock64() stamps, phase
// list run forward and then reversed. "Ops" counts 2 per MAC, matching the
// FLOP convention of the float tests, so TOPS here is directly comparable
// to the TFLOP/s there.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
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
// Kernel parameters (identical to operand_clock.hip)
// ---------------------------------------------------------------------------

constexpr int kThreads = 256;  // four wave64 warps per CTA
constexpr int kBlocks = 104;   // one CTA per CU on a 104-CU GCD
constexpr int kChains = 4;     // interleaved loop-carried accumulator chains
constexpr int kLoops = 4096;   // MFMA iterations per chain
constexpr int kRounds = 9;
constexpr int kTimedLaunches = 16;
constexpr int kWarmupSeconds = 2;

// ---------------------------------------------------------------------------
// Flavors and operand regimes
// ---------------------------------------------------------------------------

// Flavor::Large  -> v_mfma_i32_32x32x8i8, 16 accumulator lanes, K=8.
// Flavor::Small  -> v_mfma_i32_16x16x16i8, 4 accumulator lanes, K=16.
enum class Flavor { Large, Small };
enum class OperandMode { ConstantOne, ConstantMid, Random };

// Raw counter readings bracketing one wave's kernel lifetime.
struct ClockStamp {
  uint64_t core_begin, steady_begin, core_end, steady_end;
};

// Cheap per-lane hash (finalizer from the parent study): random operands
// differ per lane with no memory traffic.
__device__ uint32_t hash_mix(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  return v ^ (v >> 16);
}

// One operand byte for the given regime. Random covers the full signed
// int8 range (the analogue of "normal signed data" in a quantized world).
__device__ uint8_t operand_byte(OperandMode mode, uint32_t index) {
  switch (mode) {
    case OperandMode::ConstantOne:
      return 1;
    case OperandMode::ConstantMid:
      return 63;
    case OperandMode::Random:
      return uint8_t(hash_mix(index ^ hash_mix(index >> 16)));
  }
  return 0;
}

// ---------------------------------------------------------------------------
// The kernel: four MFMA chains, nothing else in the hot loop
// ---------------------------------------------------------------------------

using I32x16 = int32_t __attribute__((ext_vector_type(16)));
using I32x4 = int32_t __attribute__((ext_vector_type(4)));

template <Flavor F, OperandMode Mode>
__global__ __launch_bounds__(kThreads) void mfma_int8_clock_kernel(
    uint32_t* out, ClockStamp* stamps) {
  const uint64_t core_begin = clock64();
  const uint64_t steady_begin = wall_clock64();

  // Operand prologue: one packed 32-bit register holding four int8s (the
  // entire per-thread A and B operand for both flavors). For Random mode
  // the bytes come from the lane hash; otherwise they are constants.
  const uint32_t lane = threadIdx.x % 64;
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    packed |= uint32_t(operand_byte(Mode, lane * 4 + j)) << (8 * j);
  const int32_t operand = int32_t(packed);

  // Distinct chain initials keep the chains distinguishable and give exact
  // expected outputs for the constant modes.
  constexpr int kLanes = F == Flavor::Large ? 16 : 4;
  I32x16 c16[4]{};
  I32x4 c4[4]{};
#pragma unroll
  for (int chain = 0; chain < 4; ++chain)
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      if (F == Flavor::Large)
        c16[chain][j] = chain;
      else
        c4[chain][j] = chain;
    }

  // The hot loop: MFMA instructions and loop control only. Unrolling is
  // disabled so the measured rate includes the real backedge.
#pragma clang loop unroll(disable)
  for (int i = 0; i < kLoops; ++i) {
    if (F == Flavor::Large) {
      c16[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(operand, operand, c16[0],
                                                   0, 0, 0);
      c16[1] = __builtin_amdgcn_mfma_i32_32x32x8i8(operand, operand, c16[1],
                                                   0, 0, 0);
      c16[2] = __builtin_amdgcn_mfma_i32_32x32x8i8(operand, operand, c16[2],
                                                   0, 0, 0);
      c16[3] = __builtin_amdgcn_mfma_i32_32x32x8i8(operand, operand, c16[3],
                                                   0, 0, 0);
    } else {
      c4[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(operand, operand, c4[0],
                                                   0, 0, 0);
      c4[1] = __builtin_amdgcn_mfma_i32_16x16x16i8(operand, operand, c4[1],
                                                   0, 0, 0);
      c4[2] = __builtin_amdgcn_mfma_i32_16x16x16i8(operand, operand, c4[2],
                                                   0, 0, 0);
      c4[3] = __builtin_amdgcn_mfma_i32_16x16x16i8(operand, operand, c4[3],
                                                   0, 0, 0);
    }
  }

  // Reduce and store one uint32 per thread (defeats dead-code elimination).
  // Unsigned arithmetic makes the accumulator wraparound well-defined.
  uint32_t sum = 0;
#pragma unroll
  for (int chain = 0; chain < 4; ++chain)
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      if (F == Flavor::Large)
        sum += uint32_t(c16[chain][j]);
      else
        sum += uint32_t(c4[chain][j]);
    }
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

void launch_one(Flavor flavor, OperandMode mode, uint32_t* out,
                ClockStamp* stamps, hipStream_t stream) {
  const dim3 grid(kBlocks), block(kThreads);
  if (flavor == Flavor::Large) {
    switch (mode) {
      case OperandMode::ConstantOne:
        mfma_int8_clock_kernel<Flavor::Large, OperandMode::ConstantOne>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::ConstantMid:
        mfma_int8_clock_kernel<Flavor::Large, OperandMode::ConstantMid>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::Random:
        mfma_int8_clock_kernel<Flavor::Large, OperandMode::Random>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
    }
  } else {
    switch (mode) {
      case OperandMode::ConstantOne:
        mfma_int8_clock_kernel<Flavor::Small, OperandMode::ConstantOne>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::ConstantMid:
        mfma_int8_clock_kernel<Flavor::Small, OperandMode::ConstantMid>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
      case OperandMode::Random:
        mfma_int8_clock_kernel<Flavor::Small, OperandMode::Random>
            <<<grid, block, 0, stream>>>(out, stamps);
        break;
    }
  }
}

// VGPR count of a kernel instantiation, to test the "less register data"
// part of the hypothesis.
int kernel_registers(Flavor flavor) {
  hipFuncAttributes attr{};
  if (flavor == Flavor::Large)
    CHECK_HIP(hipFuncGetAttributes(
        &attr, reinterpret_cast<const void*>(
                   mfma_int8_clock_kernel<Flavor::Large,
                                          OperandMode::Random>)));
  else
    CHECK_HIP(hipFuncGetAttributes(
        &attr, reinterpret_cast<const void*>(
                   mfma_int8_clock_kernel<Flavor::Small,
                                          OperandMode::Random>)));
  return attr.numRegs;
}

// Exact expected per-thread output for constant modes, computed with the
// same modulo-2^32 arithmetic the kernel performs.
//
// Every accumulator lane gains K * v^2 per instruction (K=8 for Large,
// 16 for Small; v = operand value). After kLoops iterations the four chains
// hold pe+0 .. pe+3, so each lane contributes 4*pe+6 and the kLanes lanes
// sum to kLanes*(4*pe+6), wrapped to 32 bits.
uint32_t expected_output(Flavor flavor, OperandMode mode) {
  if (mode == OperandMode::Random) return 0;  // determinism check instead
  constexpr int kLanesArr[2] = {16, 4};       // Large, Small
  constexpr int kK[2] = {8, 16};
  const int idx = flavor == Flavor::Large ? 0 : 1;
  const uint64_t v = mode == OperandMode::ConstantOne ? 1 : 63;
  const uint64_t pe = uint64_t(kK[idx]) * v * v * kLoops;
  const uint64_t total =
      uint64_t(kLanesArr[idx]) * (4 * pe + 6);
  return uint32_t(total);  // wraps like the kernel's uint32 sum
}

// Verifies one output snapshot. Constants: exact match. Random: matched
// against the reference snapshot captured before timing (determinism).
void verify(const std::vector<uint32_t>& values, Flavor flavor,
            OperandMode mode, const std::vector<uint32_t>* reference) {
  if (mode == OperandMode::Random) {
    if (reference == nullptr) return;  // first snapshot: nothing to compare
    if (values != *reference)
      throw std::runtime_error("int8 random-mode determinism failure");
    return;
  }
  const uint32_t expected = expected_output(flavor, mode);
  for (uint32_t v : values)
    if (v != expected)
      throw std::runtime_error("int8 operand-clock kernel mismatch: got " +
                               std::to_string(v) + ", expected " +
                               std::to_string(expected));
}

// Ops (2 per MAC) in one launch, counting every wave of every CTA.
double ops_per_launch(Flavor flavor) {
  constexpr int kMacsPerInstr[2] = {32 * 32 * 8, 16 * 16 * 16};  // Large, Small
  const int idx = flavor == Flavor::Large ? 0 : 1;
  return double(kBlocks) * (kThreads / 64) * kChains * kLoops * 2.0 *
         kMacsPerInstr[idx];
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct PhaseResult {
  double median_tops;
  double median_effective_mhz;
  double median_core_cycles;
  double ops_per_core_cycle;
};

// One measurement phase: warm up, verify, timed rounds with per-wave clock
// stamps. Prints PHASE markers (unix time) for external monitor
// correlation, exactly like operand_clock.hip.
PhaseResult run_phase(Flavor flavor, OperandMode mode, const char* name,
                      int pass, uint32_t* out, ClockStamp* stamps,
                      hipStream_t stream, hipEvent_t start, hipEvent_t stop,
                      std::vector<uint32_t>& host_out,
                      std::vector<ClockStamp>& host_stamps, int steady_khz) {
  const auto launch = [&] { launch_one(flavor, mode, out, stamps, stream); };
  const double ops = ops_per_launch(flavor);

  const double unix_begin = std::chrono::duration<double>(
                                 std::chrono::system_clock::now()
                                     .time_since_epoch())
                                 .count();
  std::printf("PHASE_BEGIN pass=%d name=%s unix_s=%.3f\n", pass, name,
              unix_begin);
  std::fflush(stdout);

  // Warm-up: ~2 s of the exact same kernel.
  const auto warmup_begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 8; ++i) launch();
    CHECK_HIP(hipStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - warmup_begin <
           std::chrono::seconds(kWarmupSeconds));

  // Correctness gate before timing; snapshot for the random determinism
  // check.
  launch();
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipMemcpy(host_out.data(), out, host_out.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  const std::vector<uint32_t> reference = host_out;
  verify(host_out, flavor, mode, nullptr);

  // Timed rounds.
  std::vector<double> round_tops, wave_mhz, wave_cycles;
  for (int round = 0; round < kRounds; ++round) {
    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < kTimedLaunches; ++i) launch();
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float elapsed_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    round_tops.push_back(ops / (double(elapsed_ms) * 1000.0 / kTimedLaunches) /
                         1e6);

    // Stamps and output from the last launch of the burst.
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
    CHECK_HIP(hipMemcpy(host_out.data(), out,
                        host_out.size() * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    verify(host_out, flavor, mode, &reference);
  }

  const double unix_end = std::chrono::duration<double>(
                               std::chrono::system_clock::now()
                                   .time_since_epoch())
                               .count();
  std::printf("PHASE_END pass=%d name=%s unix_s=%.3f\n", pass, name,
              unix_end);
  std::fflush(stdout);

  PhaseResult result;
  result.median_tops = median(round_tops);
  result.median_effective_mhz = median(wave_mhz);
  result.median_core_cycles = median(wave_cycles);
  // Per-wave ops / core cycles: the clock-independent throughput measure.
  const int macs_per_instr =
      flavor == Flavor::Large ? 32 * 32 * 8 : 16 * 16 * 16;
  result.ops_per_core_cycle =
      double(kChains) * kLoops * 2.0 * macs_per_instr /
      result.median_core_cycles;
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

  // Fixed rate of wall_clock64(), to turn cycles-to-ticks ratios into
  // frequencies.
  int steady_khz = 0;
  CHECK_HIP(hipDeviceGetAttribute(&steady_khz,
                                  hipDeviceAttributeWallClockRate, 0));

  const int regs_large = kernel_registers(Flavor::Large);
  const int regs_small = kernel_registers(Flavor::Small);

  std::printf("device=0 arch=%s CUs=%d steady_clock_khz=%d\n",
              props.gcnArchName, props.multiProcessorCount, steady_khz);
  std::printf("protocol: automatic clocks, no locking, %d CTAs x %d threads, "
              "%d chains x %d loops, %d rounds x %d launches\n",
              kBlocks, kThreads, kChains, kLoops, kRounds, kTimedLaunches);
  std::printf("kernel VGPRs: 32x32x8i8=%d, 16x16x16i8=%d\n", regs_large,
              regs_small);
  std::fflush(stdout);

  // --- Runtime objects. ---
  hipStream_t stream;
  hipEvent_t start, stop;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  std::vector<uint32_t> host_out(size_t(kBlocks) * kThreads);
  std::vector<ClockStamp> host_stamps(size_t(kBlocks) * 4);
  uint32_t* out = nullptr;
  ClockStamp* stamps = nullptr;
  CHECK_HIP(hipMalloc(&out, host_out.size() * sizeof(uint32_t)));
  CHECK_HIP(hipMalloc(&stamps, host_stamps.size() * sizeof(ClockStamp)));

  // --- Phase list: both flavors x three operand regimes, forward and
  // reversed (ordering dependence would show up as a pass mismatch). ---
  struct PhaseSpec {
    Flavor flavor;
    OperandMode mode;
    const char* name;
  };
  const std::vector<PhaseSpec> phases = {
      {Flavor::Large, OperandMode::ConstantOne, "i8-32x32x8-constant-one"},
      {Flavor::Large, OperandMode::ConstantMid, "i8-32x32x8-constant-mid"},
      {Flavor::Large, OperandMode::Random, "i8-32x32x8-random"},
      {Flavor::Small, OperandMode::ConstantOne, "i8-16x16x16-constant-one"},
      {Flavor::Small, OperandMode::ConstantMid, "i8-16x16x16-constant-mid"},
      {Flavor::Small, OperandMode::Random, "i8-16x16x16-random"},
  };

  std::printf("\n%-8s %-26s %12s %12s %12s %14s\n", "pass", "phase", "TOPS",
              "eff_MHz", "core_cycles", "op/cycle");
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t i = 0; i < phases.size(); ++i) {
      const PhaseSpec& spec = pass == 0 ? phases[i]
                                        : phases[phases.size() - 1 - i];
      const PhaseResult r =
          run_phase(spec.flavor, spec.mode, spec.name, pass, out, stamps,
                    stream, start, stop, host_out, host_stamps, steady_khz);
      std::printf("%-8d %-26s %12.3f %12.2f %12.0f %14.2f\n", pass, spec.name,
                  r.median_tops, r.median_effective_mhz,
                  r.median_core_cycles, r.ops_per_core_cycle);
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
