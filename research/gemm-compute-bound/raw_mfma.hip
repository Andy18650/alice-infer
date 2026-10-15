// raw_mfma.hip — isolated "raw MFMA, no weight load" ceiling test.
//
// This is a standalone extraction of the register-MFMA reference that lives
// inside bench.hip (`reference<Matrix>` + `arithmetic_reference`). It measures
// the sustained throughput of the two native gfx90a BF16 matrix instructions
// while every operand stays in registers: no global-memory loads, no LDS
// traffic, no weight data of any kind. Whatever rate comes out is the pure
// arithmetic-issue ceiling that the real GEMM kernels in this research are
// compared against.
//
// What is measured, per (instruction flavor, grid size) configuration:
//   - v_mfma_f32_16x16x16bf16 (the "16" variant)
//   - v_mfma_f32_32x32x8bf16  (the "32" variant, used by the winning CK V3)
// launched as 104/208/416 CTAs of 256 threads (= 1/2/4 waves per CU on the
// 104 CUs of one GCD).
//
// Measurement protocol (kept from the original study): native -O3, one gfx90a
// GCD, automatic clocks, ~2 s of continuous warm-up before any timing, then
// per round one HIP-event-timed burst of 16 kernel launches. Every launch is
// verified against an exact expected value, so the compiler cannot silently
// eliminate the math.
//
// Caveat carried over from the clock audit: the operand value below is a
// constant low-magnitude BF16, and effective GFX clock depends on operand
// data, so this number is a data-dependent ceiling, not a universal constant.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
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
// Device types
// ---------------------------------------------------------------------------

// One MFMA operand per thread: four BF16 halves, matching the A/B fragment
// layout the hardware expects for both instruction flavors used here.
using Bf16x4 = uint16_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Float16 = float __attribute__((ext_vector_type(16)));

// ---------------------------------------------------------------------------
// Kernel parameters
// ---------------------------------------------------------------------------

// BF16 encoding of 2^-8 (sign=0, exponent=119, mantissa=0). All operand
// halves hold this same constant for the whole run.
constexpr uint16_t kOperandBits = 0x3b80;

// Number of independent accumulator chains. Each chain is loop-carried (its
// output feeds its own next iteration), so four interleaved chains are needed
// to cover the MFMA dependent latency and keep the matrix pipe saturated.
constexpr int kChains = 4;

// MFMA iterations per chain per kernel launch.
constexpr int kLoops = 4096;

// Threads per CTA (four wave64 warps).
constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

// Matrix == 16: v_mfma_f32_16x16x16bf16, 4 FP32 accumulator lanes per thread.
// Matrix == 32: v_mfma_f32_32x32x8bf16, 16 FP32 accumulator lanes per thread.
//
// The hot loop contains nothing but MFMA instructions and loop control: the
// operands are register constants and the accumulators never leave registers.
// Unrolling is disabled so the measured rate includes the real loop backedge
// instead of one giant unrolled block. The single store at the end defeats
// dead-code elimination.
template <int Matrix>
__global__ __launch_bounds__(kThreads) void raw_mfma_kernel(float* out) {
  const Bf16x4 operand = {kOperandBits, kOperandBits, kOperandBits,
                          kOperandBits};

  if constexpr (Matrix == 16) {
    // Distinct initial values keep the four chains distinguishable and make
    // the expected result easy to verify exactly (see expected_value()).
    Float4 c0 = {0, 0, 0, 0}, c1 = {1, 1, 1, 1}, c2 = {2, 2, 2, 2},
           c3 = {3, 3, 3, 3};
#pragma clang loop unroll(disable)
    for (int i = 0; i < kLoops; ++i) {
      c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(operand, operand, c0,
                                                      0, 0, 0);
      c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(operand, operand, c1,
                                                      0, 0, 0);
      c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(operand, operand, c2,
                                                      0, 0, 0);
      c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(operand, operand, c3,
                                                      0, 0, 0);
    }
    const Float4 total = c0 + c1 + c2 + c3;
    out[size_t(blockIdx.x) * kThreads + threadIdx.x] =
        total[0] + total[1] + total[2] + total[3];
  } else {
    Float16 c0{}, c1{}, c2{}, c3{};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      c1[j] = 1;
      c2[j] = 2;
      c3[j] = 3;
    }
#pragma clang loop unroll(disable)
    for (int i = 0; i < kLoops; ++i) {
      c0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(operand, operand, c0,
                                                    0, 0, 0);
      c1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(operand, operand, c1,
                                                    0, 0, 0);
      c2 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(operand, operand, c2,
                                                    0, 0, 0);
      c3 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(operand, operand, c3,
                                                    0, 0, 0);
    }
    const Float16 total = c0 + c1 + c2 + c3;
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) sum += total[j];
    out[size_t(blockIdx.x) * kThreads + threadIdx.x] = sum;
  }
}

// ---------------------------------------------------------------------------
// Host-side verification
// ---------------------------------------------------------------------------

// Exact expected output, per thread, for each instruction flavor.
//
// Every instruction adds sum_k(operand * operand) to each accumulator
// element: 16 * (2^-8)^2 = 2^-12 for the 16x16x16 flavor, and
// 8 * (2^-8)^2 = 2^-13 for the 32x32x8 flavor. All values are exact powers
// of two, so the FP32 arithmetic rounds never and the check can be an exact
// equality.
//
//   16x16x16: after 4096 iterations each chain gains 1.0, so the chains end
//             at 1,2,3,4; their sum is 10 per lane, and 4 lanes give 40.
//   32x32x8:  each chain gains 0.5, ending at 0.5,1.5,2.5,3.5; the sum is 8
//             per lane, and 16 lanes give 128.
float expected_value(int matrix) {
  return matrix == 16 ? 40.0f : 128.0f;
}

// Total FLOPs performed by one kernel launch. Each MFMA instruction is a
// per-wave matrix operation: MxNxK multiply-accumulate = 2*M*N*K FLOPs. One
// CTA of 256 threads holds four wave64 warps, and each warp issues
// kChains * kLoops instructions.
double flops_per_launch(int matrix, int blocks) {
  constexpr int warps_per_cta = kThreads / 64;
  const double flops_per_instruction =
      matrix == 16 ? 16.0 * 16 * 16 * 2 : 32.0 * 32 * 8 * 2;
  return double(blocks) * warps_per_cta * kChains * kLoops *
         flops_per_instruction;
}

void verify(const float* values, size_t count, int matrix) {
  const float expected = expected_value(matrix);
  for (size_t i = 0; i < count; ++i) {
    if (values[i] != expected) {
      throw std::runtime_error("MFMA reference mismatch: got " +
                               std::to_string(values[i]) + ", expected " +
                               std::to_string(expected));
    }
  }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// Time `repetitions` back-to-back launches with HIP events; returns the
// average GPU time per launch in microseconds.
template <typename Launch>
double time_gpu_us(hipStream_t stream, hipEvent_t start, hipEvent_t stop,
                   Launch launch, int repetitions) {
  CHECK_HIP(hipEventRecord(start, stream));
  for (int i = 0; i < repetitions; ++i) launch();
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));
  float elapsed_ms = 0;
  CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
  return double(elapsed_ms) * 1000.0 / repetitions;
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) try {
  int rounds = 9;  // default matches the confirmation protocol
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing argument for " + key);
    const std::string value = argv[++i];
    if (key == "--rounds") rounds = std::stoi(value);
    else throw std::runtime_error("unknown argument " + key);
  }

  // --- Device check: the instruction builtins below are gfx90a-specific. ---
  CHECK_HIP(hipSetDevice(0));
  hipDeviceProp_t props{};
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  if (std::string(props.gcnArchName).find("gfx90a") == std::string::npos ||
      props.warpSize != 64) {
    throw std::runtime_error("requires gfx90a wave64");
  }
  const int compute_units = props.multiProcessorCount;
  std::printf("device=0 arch=%s CUs=%d BF16 MFMA raw issue-rate test, "
              "no memory traffic\n",
              props.gcnArchName, compute_units);

  // --- Runtime objects: one stream, two events, one output buffer. ---
  hipStream_t stream;
  hipEvent_t start, stop;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  const int max_blocks = 416;
  std::vector<float> host_values(size_t(max_blocks) * kThreads);
  float* device_out = nullptr;
  CHECK_HIP(hipMalloc(&device_out, host_values.size() * sizeof(float)));

  // --- Warm-up: ~2 s of continuous launches to reach steady state. ---
  const auto warmup_begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 8; ++i)
      raw_mfma_kernel<32><<<208, kThreads, 0, stream>>>(device_out);
    CHECK_HIP(hipStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - warmup_begin <
           std::chrono::seconds(2));

  // --- Measurement loop: both instruction flavors x three grid sizes. ---
  // 104/208/416 CTAs on 104 CUs = 1/2/4 resident waves per CU, so the sweep
  // also shows whether occupancy limits the issue rate.
  for (int matrix : {16, 32}) {
    for (int blocks : {104, 208, 416}) {
      const auto launch_one = [&] {
        if (matrix == 16)
          raw_mfma_kernel<16><<<blocks, kThreads, 0, stream>>>(device_out);
        else
          raw_mfma_kernel<32><<<blocks, kThreads, 0, stream>>>(device_out);
      };
      // Pull the result back and verify every thread's output exactly.
      launch_one();
      CHECK_HIP(hipStreamSynchronize(stream));
      CHECK_HIP(hipMemcpy(host_values.data(), device_out,
                          size_t(blocks) * kThreads * sizeof(float),
                          hipMemcpyDeviceToHost));
      verify(host_values.data(), size_t(blocks) * kThreads, matrix);

      // Timed rounds: 16 event-timed launches per round.
      const double flops = flops_per_launch(matrix, blocks);
      std::vector<double> round_tflops;
      for (int round = 0; round < rounds; ++round) {
        const double gpu_us =
            time_gpu_us(stream, start, stop, launch_one, 16);
        round_tflops.push_back(flops / gpu_us / 1e6);
        std::printf("matrix=%dx%dx%dbf16 blocks=%d round=%d tflops=%.3f "
                    "gpu_us=%.3f\n",
                    matrix, matrix, matrix == 16 ? 16 : 8, blocks, round,
                    round_tflops.back(), gpu_us);
      }
      std::sort(round_tflops.begin(), round_tflops.end());
      const double median = round_tflops[round_tflops.size() / 2];
      std::printf("SUMMARY matrix=%dx%dx%dbf16 blocks=%d median_tflops=%.3f "
                  "over %d rounds\n\n",
                  matrix, matrix, matrix == 16 ? 16 : 8, blocks, median,
                  rounds);
    }
  }

  CHECK_HIP(hipFree(device_out));
  CHECK_HIP(hipEventDestroy(stop));
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipStreamDestroy(stream));
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return 1;
}
