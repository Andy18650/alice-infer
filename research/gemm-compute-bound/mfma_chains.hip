// mfma_chains.hip — how many independent MFMA accumulator chains are needed
// to reach peak matrix-unit throughput?
//
// Follow-up (user direction): the same sweep on v_mfma_f32_16x16x16bf16 —
// 4 accumulator lanes per thread (vs 16) and half the MACs per instruction
// (4096 vs 8192), so the pipe occupancy is T = 32 cycles (vs 64). If the
// dependent latency also scales with the instruction (L ~ T), one chain
// saturates this flavor too; if L is fixed by writeback depth independent
// of instruction size, the 16x16x16 flavor may need proportionally more
// chains. Both flavors run in this binary; the 32x32x8 results reproduce
// the earlier single-flavor experiment as a control.
//
// Every hand-rolled kernel in the clock studies (and the CK GEMM itself)
// uses multiple independent accumulator chains to cover MFMA dependent
// latency. This experiment measures the minimum: the reference loop re-run
// with Chains in {1, 2, 3, 4, 8} independent accumulators.
//
// Theory being tested: with C chains, a wave issues MFMA instructions at
// max(T, L/C) cycles each, where
//   T = pipe occupancy: 64 cycles per v_mfma_f32_32x32x8bf16, 32 cycles
//       per v_mfma_f32_16x16x16bf16 (both 128 MACs/cycle/SIMD)
//   L = dependent latency of a chained MFMA (result -> next instruction
//       on the same accumulator), never isolated in our records.
// So: if the matrix unit forwards accumulators internally (L ~ T), one
// chain reaches the full ~255 flop/core-cycle. The known 4-chain point
// (255.3 flop/cycle on 32x32x8) only bounds L <= ~257 cycles.
//
// Setup is deliberately the same as raw_mfma.hip: one wave per SIMD (104
// CTAs x 4 waves; each SIMD's matrix unit sees exactly one wave, so
// nothing hides per-wave latency except the wave's own chains), register
// constant operands, no memory traffic in the hot loop, per-wave
// clock64()/wall_clock64() stamps for the clock-independent flop/cycle
// metric.
//
// Verification: chain i starts at i; per instruction every lane gains
// K * (2^-8)^2 (K = 16 resp. 8), all powers of two, so after kLoops
// iterations the expected output is exact:
//   16x16x16: chain ends at i + 1.0; per-thread sum = 4 * sum(i+1)
//   32x32x8:  chain ends at i + 0.5; per-thread sum = 16 * sum(i+0.5).

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

constexpr int kThreads = 256;  // four wave64 warps per CTA
constexpr int kBlocks = 104;   // one CTA per CU; one wave per SIMD
constexpr int kLoops = 4096;   // iterations per chain
constexpr int kRounds = 9;
constexpr int kTimedLaunches = 16;
constexpr int kWarmupSeconds = 2;

using Bf16x4 = uint16_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Float16v = float __attribute__((ext_vector_type(16)));

// Raw counter readings bracketing one wave's kernel lifetime.
struct ClockStamp {
  uint64_t core_begin, steady_begin, core_end, steady_end;
};

constexpr int kLanes(int matrix) { return matrix == 16 ? 4 : 16; }
constexpr int kMacsPerInstr(int matrix) { return matrix == 16 ? 16 * 16 * 16 : 32 * 32 * 8; }

template <int Matrix, int Chains>
__global__ __launch_bounds__(kThreads) void mfma_chains_kernel(
    float* out, ClockStamp* stamps) {
  const uint64_t core_begin = clock64();
  const uint64_t steady_begin = wall_clock64();

  // Register-constant BF16 operand 2^-8 — no memory traffic anywhere.
  const Bf16x4 operand = {0x3b80, 0x3b80, 0x3b80, 0x3b80};

  // Chains independent accumulators; chain i initialized to i so the
  // expected result is exactly checkable per chain count.
  Float16v c16[Chains];
  Float4 c4[Chains];
#pragma unroll
  for (int i = 0; i < Chains; ++i)
#pragma unroll
    for (int j = 0; j < kLanes(Matrix); ++j) {
      if (Matrix == 16)
        c4[i][j] = float(i);
      else
        c16[i][j] = float(i);
    }

  // The hot loop: each chain's next instruction depends only on its own
  // previous result. With Chains == 1 this is a single dependent MFMA
  // chain — the case under test. Unrolling is disabled so the measured
  // rate includes the real loop backedge.
#pragma clang loop unroll(disable)
  for (int i = 0; i < kLoops; ++i) {
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      if (Matrix == 16)
        c4[chain] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
            operand, operand, c4[chain], 0, 0, 0);
      else
        c16[chain] = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
            operand, operand, c16[chain], 0, 0, 0);
    }
  }

  // Reduce all chains and lanes into one float per thread (defeats
  // dead-code elimination).
  float sum = 0;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain)
#pragma unroll
    for (int j = 0; j < kLanes(Matrix); ++j) {
      if (Matrix == 16)
        sum += c4[chain][j];
      else
        sum += c16[chain][j];
    }
  out[size_t(blockIdx.x) * kThreads + threadIdx.x] = sum;

  // Per-wave timestamps, recorded after the hot loop.
  const uint64_t steady_end = wall_clock64();
  const uint64_t core_end = clock64();
  if (threadIdx.x % 64 == 0)
    stamps[size_t(blockIdx.x) * 4 + threadIdx.x / 64] = {
        core_begin, steady_begin, core_end, steady_end};
}

// Exact expected per-thread output for a given flavor and chain count
// (see header).
float expected_output(int matrix, int chains) {
  const double gain = matrix == 16 ? 1.0 : 0.5;  // per lane per chain
  const double per_chain_total =
      double(chains) * (chains - 1) / 2.0 + gain * chains;
  return float(kLanes(matrix) * per_chain_total);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <int Matrix, int Chains>
void run_chains(const char* name, float* out, ClockStamp* stamps,
                hipStream_t stream, hipEvent_t start, hipEvent_t stop,
                std::vector<float>& host_out,
                std::vector<ClockStamp>& host_stamps, int steady_khz) {
  const auto launch = [&] {
    mfma_chains_kernel<Matrix, Chains>
        <<<kBlocks, kThreads, 0, stream>>>(out, stamps);
  };
  // FLOPs per launch: CTAs * warps * chains * loops * (2 * M * N * K).
  const double flops =
      double(kBlocks) * (kThreads / 64) * Chains * kLoops * 2.0 *
      kMacsPerInstr(Matrix);

  // Warm-up: ~2 s of the exact same kernel.
  const auto warmup_begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 8; ++i) launch();
    CHECK_HIP(hipStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - warmup_begin <
           std::chrono::seconds(kWarmupSeconds));

  // Correctness gate.
  launch();
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipMemcpy(host_out.data(), out, host_out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const float expected = expected_output(Matrix, Chains);
  for (float v : host_out)
    if (v != expected)
      throw std::runtime_error(std::string(name) + ": mismatch, got " +
                               std::to_string(v) + " expected " +
                               std::to_string(expected));

  // Timed rounds with per-wave stamps.
  std::vector<double> round_tflops, wave_mhz, wave_cycles;
  for (int round = 0; round < kRounds; ++round) {
    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < kTimedLaunches; ++i) launch();
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    round_tflops.push_back(flops / (double(ms) * 1000.0 / kTimedLaunches) /
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

  const double cyc = median(wave_cycles);
  const double cycles_per_instr = cyc / (double(Chains) * kLoops);
  const double flop_per_cycle =
      double(Chains) * kLoops * 2.0 * kMacsPerInstr(Matrix) / cyc;
  std::printf("%-6s %s chains=%d %10.3f TFLOP/s  eff %7.2f MHz  "
              "%7.2f cyc/instr  %7.2f flop/cyc\n",
              name, Matrix == 16 ? "16x16x16" : "32x32x8 ", Chains,
              median(round_tflops), median(wave_mhz), cycles_per_instr,
              flop_per_cycle);
  std::fflush(stdout);
}

}  // namespace

int main() try {
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
  std::printf("one wave per SIMD, %d CTAs x %d threads, %d loops, "
              "%d rounds x %d launches\n",
              kBlocks, kThreads, kLoops, kRounds, kTimedLaunches);
  std::fflush(stdout);

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

  // The new question first (16x16x16), then the 32x32x8 control; each
  // flavor ascending then descending, to check ordering effects.
  run_chains<16, 1>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 2>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 3>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 4>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 8>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 8>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 4>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 2>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<16, 1>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 1>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 2>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 3>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 4>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 8>("pass0", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 8>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 4>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 2>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);
  run_chains<32, 1>("pass1", out, stamps, stream, start, stop, host_out,
                    host_stamps, steady_khz);

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
