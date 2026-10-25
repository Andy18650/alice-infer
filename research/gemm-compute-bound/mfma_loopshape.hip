// mfma_loopshape.hip — what exactly costs the extra ~12.5 cycles when a
// lone dependent MFMA is the only work in a loop body?
//
// Background (mfma_chains.hip + the parallel study in temp/mfma-dependency):
//   - 8 dependent MFMAs per loop backedge (same accumulator!): 32.02
//     cyc/instr — the matrix unit forwards the accumulator internally and
//     a single chain runs at full pipe rate T=32 (16x16x16) / T=64
//     (32x32x8).
//   - 1 dependent MFMA per backedge (16x16x16, AGPR accumulator): 44.5
//     cyc/instr. For 32x32x8 the penalty is invisible (64.6 vs T=64).
//   - 2+ MFMAs per backedge: back to ~32.1.
// All observations fit per-backedge cost = max(n*T, ~44.5). The remaining
// question is WHAT the ~44.5 is. Two candidate mechanisms:
//   writeback theory: a fixed ~44.5-cycle accumulator writeback latency
//     that overlaps subsequent pipe occupancy. A consumer that cannot use
//     the internal forwarding path (it is after the backedge, not in the
//     producer's basic block) waits for the full writeback; the wait is
//     exposed only when the pipe work before the consumer is < 44.5.
//   branch-cost theory: the backedge itself (fetch/issue realignment of
//     the lone MFMA after the branch) costs ~12.5 cycles whenever the
//     loop body is a single MFMA, DEPENDENCY OR NOT.
// The theories diverge on exactly one cleanly implementable shape:
//   nodep: ONE MFMA per backedge, C input is a loop-varying constant —
//     no read-after-write dependency at all (only a write-after-write on
//     the destination). Writeback theory -> ~32 cyc/instr (nobody waits
//     for the writeback); branch-cost theory -> ~44.5.
// dep1 (the known lone-dependent-MFMA loop) runs as a control in both
// accumulator flavors: the intrinsic AGPR form must reproduce 44.5, and
// the inline-asm VGPR form checks that the phenomenon is not
// AGPR-specific (the parallel study only verified flavor equivalence for
// same-block chains).
//
// A dep2 shape (consumer two backedges away, 64 cycles of slack) was
// considered and dropped: any in-loop slack-based variant is
// operationally equivalent under both theories (both predict
// max(T, W)), and implementing it safely is not possible — an in-loop
// VALU select of the accumulator would read the MFMA result ~14 issue
// slots after issue, before the ~44-cycle VGPR writeback lands (gfx90a
// has no hardware interlock for inline-asm MFMA -> VALU reads; the
// parallel study hit the same hazard and added settle()/bridge code).
//
// The asm-based shapes therefore borrow the parallel study's exit
// lesson: a volatile "+v" (read-write) barrier over the result
// registers, with the conservative s_nop drain inside it, keeps every
// later VALU read behind the writeback.
//
// Same protocol as mfma_chains.hip: native -O3, one wave per SIMD
// (104 CTAs x 256 threads), 2 s warm-up, 9 rounds x 16 event-timed
// launches, per-wave clock64()/wall_clock64() stamps, forward+reverse
// pass order. Verification values are exact in fp32 (C input uses only
// 0/1 and the operand is 2^-8, so every partial sum is a power-of-two
// combination).

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
constexpr int kLoops = 4096;   // MFMA instructions per wave (one per backedge)
constexpr int kRounds = 9;
constexpr int kTimedLaunches = 16;
constexpr int kWarmupSeconds = 2;

using Bf16x4 = uint16_t __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
using Float16v = float __attribute__((ext_vector_type(16)));

struct ClockStamp {
  uint64_t core_begin, steady_begin, core_end, steady_end;
};

constexpr int kLanes(int matrix) { return matrix == 16 ? 4 : 16; }
constexpr int kMacsPerInstr(int matrix) {
  return matrix == 16 ? 16 * 16 * 16 : 32 * 32 * 8;
}

// One 16x16x16 MFMA with the accumulator in VGPRs (asm flavor; the "_1k"
// XDL encoding accepts VGPR C/D operands on gfx90a — the parallel study
// ran this exact form at the same 32.02 cyc/instr as the intrinsic).
// In-place form: the destination is allowed (and for chained use,
// desired) to share its register with the C operand.
__device__ __forceinline__ Float4 mfma16_vgpr(Bf16x4 a, Bf16x4 b, Float4 c) {
  Float4 d;
  asm volatile("v_mfma_f32_16x16x16bf16_1k %0, %1, %2, %3"
               : "=v"(d)
               : "v"(a), "v"(b), "v"(c));
  return d;
}

// Shape 0 (dep1_agpr):  intrinsic AGPR chain — the known 44.5 case.
// Shape 1 (dep1_vgpr):  asm VGPR chain — flavor control for shape 0.
// Shape 3 (nodep_vgpr): single destination, loop-varying constant C input
//                       (no RAW; cross-backedge WAW on the destination).
// Shape 5 (nodep_agpr): same as 3 via the intrinsic (compiler-managed
//                       hazards, AGPR destination).
// Every shape issues exactly ONE MFMA per loop backedge.
template <int Matrix, int Shape>
__global__ __launch_bounds__(kThreads) void loop_shape_kernel(
    float* out, ClockStamp* stamps) {
  const uint64_t core_begin = clock64();
  const uint64_t steady_begin = wall_clock64();

  const Bf16x4 operand = {0x3b80, 0x3b80, 0x3b80, 0x3b80};
  Float4 a4[2];
  Float16v a16[2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    a4[0][j] = 0.0f;
    a4[1][j] = 1.0f;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    a16[0][j] = 0.0f;
    a16[1][j] = 1.0f;
  }
  // nodep C input: invariant 2.0 (distinct from every other constant in
  // the kernel, so verification catches any accidental chaining).
  Float4 cin = {2.0f, 2.0f, 2.0f, 2.0f};
  // nodep destination: written by the MFMA each iteration, read only in
  // the exit reduction.
  Float4 dst = {};

#pragma clang loop unroll(disable)
  for (int i = 0; i < kLoops; ++i) {
    if (Shape == 0) {
      if (Matrix == 16)
        a4[0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(operand, operand,
                                                          a4[0], 0, 0, 0);
      else
        a16[0] = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(operand, operand,
                                                          a16[0], 0, 0, 0);
    } else if (Shape == 1) {
      a4[0] = mfma16_vgpr(operand, operand, a4[0]);
    } else if (Shape == 3) {
      // nodep: C input is a loop-invariant constant — no read-after-write
      // dependency at all. The asm is written directly (no helper, no
      // assignment of its result) so the destination register is written
      // by the MFMA and read ONLY inside the exit asm: any in-loop read
      // (including compiler-inserted copies of the result, seen in an
      // earlier build) stalls the loop on the writeback scoreboard and
      // reintroduces exactly the latency this shape is meant to remove.
      // The C operand is "+v" (in-out) so it stays live without being
      // assignable from the result, and the destination is earlyclobber
      // so the two can never share a register.
      asm volatile("v_mfma_f32_16x16x16bf16_1k %0, %2, %3, %1"
                   : "=&v"(dst), "+v"(cin)
                   : "v"(operand), "v"(operand));
    } else {  // Shape == 5
      // nodep via the intrinsic, C input loop-invariant so the
      // accumulator materialization (v_accvgpr_write of the constant)
      // hoists out of the loop — the varying-C form measured 88.5
      // cyc/instr, dominated by per-iteration accvgpr_write/MFMA
      // interference, not by the phenomenon under study.
      a4[0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(operand, operand,
                                                        cin, 0, 0, 0);
    }
  }

  // Exit reduction for the asm shapes, entirely inside one volatile
  // asm block: inline-asm MFMA -> VALU read hazards are not
  // compiler-managed on gfx90a (the parallel study hit this too), and
  // neither a plain s_nop drain nor a "+v" barrier suffices — the
  // scheduler hoists the VALU reads (or copies of them) above it. With
  // the reads inside the block, nothing can move. The untouched acc1
  // (4 lanes x 1.0) is folded in as the constant 4.0 — the compiler
  // constant-folded it identically in earlier builds.
  float sum;
  if (Shape == 1) {
    const float acc_one_sum = 4.0f;  // untouched acc1: 4 lanes x 1.0
    asm volatile(
        "s_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_nop 15\n\t"
        "s_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_nop 15\n\t"
        "v_add_f32_e32 %0, 0, %1\n\t"
        "v_add_f32_e32 %0, %0, %2\n\t"
        "v_add_f32_e32 %0, %0, %3\n\t"
        "v_add_f32_e32 %0, %0, %4\n\t"
        "v_add_f32_e32 %0, %0, %5"
        : "=&v"(sum)
        : "v"(a4[0][0]), "v"(a4[0][1]), "v"(a4[0][2]), "v"(a4[0][3]),
          "v"(acc_one_sum)
        : "memory");
  } else if (Shape == 3) {
    const float acc_one_sum = 4.0f;
    asm volatile(
        "s_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_nop 15\n\t"
        "s_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_nop 15\n\t"
        "v_add_f32_e32 %0, 0, %1\n\t"
        "v_add_f32_e32 %0, %0, %2\n\t"
        "v_add_f32_e32 %0, %0, %3\n\t"
        "v_add_f32_e32 %0, %0, %4\n\t"
        "v_add_f32_e32 %0, %0, %5\n\t"
        "v_add_f32_e32 %0, %0, %6\n\t"
        "v_add_f32_e32 %0, %0, %7\n\t"
        "v_add_f32_e32 %0, %0, %8\n\t"
        "v_add_f32_e32 %0, %0, %9"
        : "=&v"(sum)
        : "v"(dst[0]), "v"(dst[1]), "v"(dst[2]), "v"(dst[3]),
          "v"(cin[0]), "v"(cin[1]), "v"(cin[2]), "v"(cin[3]),
          "v"(acc_one_sum)
        : "memory");
  } else {
    sum = 0;
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int j = 0; j < kLanes(Matrix); ++j) {
        if (Matrix == 16)
          sum += a4[c][j];
        else
          sum += a16[c][j];
      }
  }
  out[size_t(blockIdx.x) * kThreads + threadIdx.x] = sum;

  const uint64_t steady_end = wall_clock64();
  const uint64_t core_end = clock64();
  if (threadIdx.x % 64 == 0)
    stamps[size_t(blockIdx.x) * 4 + threadIdx.x / 64] = {
        core_begin, steady_begin, core_end, steady_end};
}

// Exact expected per-thread outputs. Per-instruction lane gain:
// 16x16x16 -> 16*(2^-8)^2 = 2^-12; 32x32x8 -> 8*(2^-8)^2 = 2^-13.
float expected_output(int matrix, int shape) {
  const double gain = matrix == 16 ? 1.0 / 4096.0 : 1.0 / 8192.0;
  if (matrix == 32)  // only shape 0 runs for 32x32x8; acc1 keeps its 1.0 init
    return float(16.0 * (1.0 + kLoops * gain));
  switch (shape) {
    case 0:  // dep1: acc0 = 0 + kLoops*gain; acc1 keeps its 1.0 init
    case 1:
      return float(4.0 * (1.0 + kLoops * gain));
    case 3:  // nodep: dst = cin + gain per lane; + cin lanes + acc1 lanes
      return float(4.0 * (2.0 + gain) + 4.0 * 2.0 + 4.0 * 1.0);
    case 5:  // nodep via intrinsic: dst = cin + gain; acc1 keeps 1.0 init
      return float(4.0 * (2.0 + gain) + 4.0 * 1.0);
    default:
      throw std::logic_error("unexpected shape");
  }
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

const char* shape_name(int shape) {
  switch (shape) {
    case 0: return "dep1_agpr ";
    case 1: return "dep1_vgpr ";
    case 3: return "nodep_vgpr";
    case 5: return "nodep_agpr";
    default: return "??        ";
  }
}

template <int Matrix, int Shape>
void run_shape(float* out, ClockStamp* stamps, hipStream_t stream,
               hipEvent_t start, hipEvent_t stop, std::vector<float>& host_out,
               std::vector<ClockStamp>& host_stamps, int steady_khz) {
  const auto launch = [&] {
    loop_shape_kernel<Matrix, Shape>
        <<<kBlocks, kThreads, 0, stream>>>(out, stamps);
  };
  const double flops = double(kBlocks) * (kThreads / 64) * kLoops * 2.0 *
                       kMacsPerInstr(Matrix);

  const auto warmup_begin = std::chrono::steady_clock::now();
  do {
    for (int i = 0; i < 8; ++i) launch();
    CHECK_HIP(hipStreamSynchronize(stream));
  } while (std::chrono::steady_clock::now() - warmup_begin <
           std::chrono::seconds(kWarmupSeconds));

  launch();
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipMemcpy(host_out.data(), out, host_out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  const float expected = expected_output(Matrix, Shape);
  for (float v : host_out)
    if (v != expected)
      throw std::runtime_error(std::string(shape_name(Shape)) +
                               ": mismatch, got " + std::to_string(v) +
                               " expected " + std::to_string(expected));

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
  const double cycles_per_instr = cyc / double(kLoops);
  const double flop_per_cycle =
      double(kLoops) * 2.0 * kMacsPerInstr(Matrix) / cyc;
  std::printf("%s %s %10.3f TFLOP/s  eff %7.2f MHz  %7.2f cyc/instr  "
              "%7.2f flop/cyc\n",
              Matrix == 16 ? "16x16x16" : "32x32x8 ", shape_name(Shape),
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
  std::printf("one wave per SIMD, %d CTAs x %d threads, %d one-MFMA "
              "iterations, %d rounds x %d launches\n",
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

  // 16x16x16 (where the penalty is visible), forward + reverse.
  run_shape<16, 0>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 1>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 3>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 5>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 5>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 3>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 1>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<16, 0>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  // 32x32x8 control: dep1 must stay ~64.5 (penalty hidden under T=64).
  run_shape<32, 0>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);
  run_shape<32, 0>(out, stamps, stream, start, stop, host_out, host_stamps,
                   steady_khz);

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
