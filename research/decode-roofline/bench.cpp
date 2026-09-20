// Decode roofline decomposition for DeepSeek-R1-Distill-Llama-8B on one
// gfx90a GCD (research, user-directed 2026-09). The engine decodes at
// 34.2 tok/s (29.28 ms/token) against a weights-traffic roofline of
// ~106 tok/s (15.01 GB/token @ 1.6 TB/s). This bench explains the gap
// WITHOUT touching the engine: it links the engine's real kernel objects
// (build*/alice_gpu_kernels.o + gpu_rocm_kernels.o) and rocBLAS, replicates
// the exact decode-step kernel sequence at S=1, and decomposes per-token
// time into GEMM / SIMT / gap / sync components. gemv_ref.hip adds a
// pure-read ceiling and a simple split-K bf16 GEMV as the "what a proper
// GEMV could do" reference.
//
// Phases:
//   [1] HBM streaming-read ceiling (1 GiB)
//   [2] per-shape S=1 GEMV: rocBLAS gemm_ex (the engine's exact call)
//       vs the hand-rolled split-K GEMV; per-token GEMM floors
//   [3] SIMT-only chain (engine kernels, exact decode sequence, no GEMMs)
//   [4] full replica: async / per-token-sync / CPU-enqueue-rate
//   [5] phase breakdown via hipEvents (attn blocks / ffn blocks / tail)
//
// Usage: bench_decode [device]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

// The engine's real kernel wrappers (host side, raw pointers).
#include "../../src/blocks/gpu_rocm/kernels.hpp"
#include "../../src/gpu/kernels.hpp"

// gemv_ref.hip host wrappers.
namespace decode_roofline {
void read_bw(const void* w, size_t bytes, float* sink);
void gemv_splitk(const void* W, const void* x, float* y, float* partials, uint32_t N,
                 uint32_t K, uint32_t CK);
}

namespace {

#define HIP_CHECK(x)                                                                \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      std::fprintf(stderr, "HIP error %d at %s:%d\n", int(e_), __FILE__, __LINE__); \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)
#define RB_CHECK(x)                                                                 \
  do {                                                                              \
    rocblas_status s_ = (x);                                                        \
    if (s_ != rocblas_status_success) {                                             \
      std::fprintf(stderr, "rocBLAS error %d at %s:%d\n", int(s_), __FILE__,        \
                   __LINE__);                                                       \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)

namespace gk = alice::gpu::kernels;
namespace rk = alice::gpu_rocm::kernels;

using Clock = std::chrono::steady_clock;

// --- 8B model constants (models/deepseek-r1-distill-llama-8b.manifest.json)
constexpr uint32_t kD = 4096, kInter = 14336, kLayers = 32;
constexpr uint32_t kNh = 32, kNkv = 8, kHd = 128, kV = 128256;
constexpr uint32_t kQkvN = (kNh + 2 * kNkv) * kHd;  // 6144
constexpr uint32_t kQd = kNh * kHd;                 // 4096
constexpr float kScale = 0.08838834764831845f;      // 1/sqrt(128)
constexpr uint32_t kMaxT = 1024;                    // decode context cap here

struct Shape {
  const char* name;
  uint32_t N, K;
  uint32_t per_token;  // calls per decode token
};
const Shape kShapes[] = {
    {"attn qkv  [ 6144 x  4096]", kQkvN, kD, kLayers},
    {"attn o    [ 4096 x  4096]", kD, kQd, kLayers},
    {"ffn gup   [28672 x  4096]", 2 * kInter, kD, kLayers},
    {"ffn down  [ 4096 x 14336]", kD, kInter, kLayers},
    {"lm_head   [128256 x  4096]", kV, kD, 1},
};

// The engine's S=1 GEMM: rocBLAS col-major view, C'[N,1] = W.x with W
// row-major [N,K] bf16 read as col-major [K,N] via opT (lda=K), x bf16
// [1,K] (ldb=K), y f32 [N,1] (ldc=N). Identical call shape to
// GqaAttentionBF16/DenseFfnBF16/LmHead forwards.
void gemm_s1(rocblas_handle h, const void* W, const void* x, float* y, uint32_t N,
             uint32_t K) {
  const float alpha = 1.0f, beta = 0.0f;
  RB_CHECK(rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, 1,
                           K, &alpha, W, rocblas_datatype_bf16_r, K, x,
                           rocblas_datatype_bf16_r, K, &beta, y, rocblas_datatype_f32_r,
                           N, y, rocblas_datatype_f32_r, N, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));
}

// Mathematically identical, different rocBLAS view: y^T[1,N] = x^T[1,K]
// . W^T[K,N] with M=1 ("wide-N" row-output form). Selects different Tensile
// kernels — probed to see whether the engine's TN form is the problem.
void gemm_s1_wide(rocblas_handle h, const void* W, const void* x, float* y, uint32_t N,
                  uint32_t K) {
  const float alpha = 1.0f, beta = 0.0f;
  RB_CHECK(rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, 1,
                           N, K, &alpha, x, rocblas_datatype_bf16_r, K, W,
                           rocblas_datatype_bf16_r, K, &beta, y, rocblas_datatype_f32_r,
                           1, y, rocblas_datatype_f32_r, 1, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));
}

// Event-timed loop: fn enqueued `iters` times back-to-back on the default
// stream (no per-iter sync), one event pair around the whole batch.
double bench_ms(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  HIP_CHECK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return double(ms) / iters;
}

void* dmalloc(size_t bytes) {
  void* p = nullptr;
  HIP_CHECK(hipMalloc(&p, bytes));
  return p;
}
// bf16 device buffer filled with 0x3F bytes -> every element 0x3F3F (~0.748):
// sane finite values, no denormals/NaNs, no host staging needed.
void dbf16_fill(unsigned short* p, size_t bytes) { HIP_CHECK(hipMemset(p, 0x3F, bytes)); }

// --- the replica: every device buffer one decode token touches ---------------
struct Replica {
  unsigned short *w_qkv, *w_o, *w_gup, *w_down, *w_lm;  // weights (bf16)
  float *residual, *io, *gamma;                          // trunk [4096]
  unsigned short *dx, *dattn, *dact;                     // bf16 gemm operands
  float *dqkv, *dgu, *logits;                            // f32 gemm outputs
  float *kcache, *vcache, *cos_t, *sin_t, *dsplit;       // attention state
  float* dpart;                                          // split-K partials (ref gemv)
  int32_t *ids_dev, *did;
  int32_t id_host = 0;
};

enum Mode { kGemm = 1, kSimt = 2, kFull = 3, kWithD2H = 4, kRefGemm = 8 };

// Dispatch one GEMV site according to mode (rocBLAS engine form vs the
// hand-rolled split-K reference).
inline void gemv_site(Replica& r, rocblas_handle h, unsigned mode, const void* W,
                      const void* x, float* y, uint32_t N, uint32_t K) {
  if (mode & kRefGemm)
    decode_roofline::gemv_splitk(W, x, y, r.dpart, N, K, 2048);
  else
    gemm_s1(h, W, x, y, N, K);
}

// One decode token, exactly the engine's kernel sequence at S=1:
// Embedder::embed (H2D ids + gather) -> per layer [Trunk::read rmsnorm,
// GqaAttention (convert, gemm, rope, attention x2, gemm), Trunk::write,
// rmsnorm, DenseFfn (convert, gemm, silu_mul, gemm), Trunk::write] ->
// Trunk::output rmsnorm -> LmHead (convert, gemm) -> argmax -> D2H id.
void token(Replica& r, rocblas_handle h, uint32_t pos, unsigned mode) {
  const bool gemm = mode & (kGemm | kRefGemm), simt = mode & kSimt;
  if (simt) {
    HIP_CHECK(hipMemcpy(r.ids_dev, &r.id_host, sizeof(int32_t), hipMemcpyHostToDevice));
    gk::embed_gather(r.w_lm, r.ids_dev, r.residual, 1, kD);
  }
  for (uint32_t l = 0; l < kLayers; ++l) {
    // --- attention block
    if (simt) gk::rmsnorm(r.residual, r.gamma, r.io, 1, kD, 1e-5f);
    if (simt) gk::f32_to_bf16(r.io, r.dx, kD);
    if (gemm) gemv_site(r, h, mode, r.w_qkv, r.dx, r.dqkv, kQkvN, kD);
    if (simt)
      rk::rope_append(r.dqkv, r.kcache, r.vcache, r.cos_t, r.sin_t, pos, 1, kNh, kNkv,
                      kHd);
    if (simt)
      rk::attention(r.dqkv, r.kcache, r.vcache, r.dattn, r.dsplit, nullptr, pos, 1, kNh,
                    kNkv, kHd, kScale);
    if (gemm) gemv_site(r, h, mode, r.w_o, r.dattn, r.io, kD, kQd);
    if (simt) gk::add_rows(r.residual, r.io, kD);
    // --- ffn block
    if (simt) gk::rmsnorm(r.residual, r.gamma, r.io, 1, kD, 1e-5f);
    if (simt) gk::f32_to_bf16(r.io, r.dx, kD);
    if (gemm) gemv_site(r, h, mode, r.w_gup, r.dx, r.dgu, 2 * kInter, kD);
    if (simt) rk::silu_mul(r.dgu, r.dact, 1, kInter);
    if (gemm) gemv_site(r, h, mode, r.w_down, r.dact, r.io, kD, kInter);
    if (simt) gk::add_rows(r.residual, r.io, kD);
  }
  if (simt) gk::rmsnorm(r.residual, r.gamma, r.io, 1, kD, 1e-5f);  // trunk output
  if (simt) gk::f32_to_bf16(r.io, r.dx, kD);                       // lm_head convert
  if (gemm) gemv_site(r, h, mode, r.w_lm, r.dx, r.logits, kV, kD);
  if (mode & kWithD2H) {
    if (simt) gk::argmax(r.logits, kV, r.did);
    HIP_CHECK(hipMemcpy(&r.id_host, r.did, sizeof(int32_t), hipMemcpyDeviceToHost));
  }
}

// Phase-split token for hipEvent breakdown: runs the full sequence and
// records events around the attention part / ffn part of each layer.
struct PhaseTimers {
  std::vector<hipEvent_t> ev_attn_a, ev_attn_b, ev_ffn_a, ev_ffn_b;
  hipEvent_t tail_a = nullptr, tail_b = nullptr;
};

void token_phased(Replica& r, rocblas_handle h, uint32_t pos, PhaseTimers& t) {
  HIP_CHECK(hipMemcpy(r.ids_dev, &r.id_host, sizeof(int32_t), hipMemcpyHostToDevice));
  gk::embed_gather(r.w_lm, r.ids_dev, r.residual, 1, kD);
  for (uint32_t l = 0; l < kLayers; ++l) {
    gk::rmsnorm(r.residual, r.gamma, r.io, 1, kD, 1e-5f);
    gk::f32_to_bf16(r.io, r.dx, kD);
    HIP_CHECK(hipEventRecord(t.ev_attn_a[l]));
    gemm_s1(h, r.w_qkv, r.dx, r.dqkv, kQkvN, kD);
    rk::rope_append(r.dqkv, r.kcache, r.vcache, r.cos_t, r.sin_t, pos, 1, kNh, kNkv,
                    kHd);
    rk::attention(r.dqkv, r.kcache, r.vcache, r.dattn, r.dsplit, nullptr, pos, 1, kNh,
                  kNkv, kHd, kScale);
    gemm_s1(h, r.w_o, r.dattn, r.io, kD, kQd);
    HIP_CHECK(hipEventRecord(t.ev_attn_b[l]));
    gk::add_rows(r.residual, r.io, kD);
    gk::rmsnorm(r.residual, r.gamma, r.io, 1, kD, 1e-5f);
    gk::f32_to_bf16(r.io, r.dx, kD);
    HIP_CHECK(hipEventRecord(t.ev_ffn_a[l]));
    gemm_s1(h, r.w_gup, r.dx, r.dgu, 2 * kInter, kD);
    rk::silu_mul(r.dgu, r.dact, 1, kInter);
    gemm_s1(h, r.w_down, r.dact, r.io, kD, kInter);
    HIP_CHECK(hipEventRecord(t.ev_ffn_b[l]));
    gk::add_rows(r.residual, r.io, kD);
  }
  gk::rmsnorm(r.residual, r.gamma, r.io, 1, kD, 1e-5f);
  gk::f32_to_bf16(r.io, r.dx, kD);
  HIP_CHECK(hipEventRecord(t.tail_a));
  gemm_s1(h, r.w_lm, r.dx, r.logits, kV, kD);
  gk::argmax(r.logits, kV, r.did);
  HIP_CHECK(hipEventRecord(t.tail_b));
}

}  // namespace

int main(int argc, char** argv) {
  const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  char ver[128];
  rocblas_get_version_string(ver, sizeof(ver));
  std::printf("== device %d: %s (%s), %d CUs | rocBLAS %s ==\n\n", dev, prop.name,
              prop.gcnArchName, prop.multiProcessorCount, ver);

  rocblas_handle handle;
  RB_CHECK(rocblas_create_handle(&handle));

  // ------------------------------------------------------------------
  std::printf("[1] HBM streaming-read ceiling\n");
  {
    const size_t bytes = size_t(1) << 30;  // 1 GiB
    unsigned short* w = (unsigned short*)dmalloc(bytes);
    float* sink = (float*)dmalloc(1024);
    dbf16_fill(w, bytes);
    const double ms = bench_ms([&] { decode_roofline::read_bw(w, bytes, sink); }, 3, 20);
    std::printf("    1.00 GiB in %6.3f ms = %5.0f GB/s\n\n", ms,
                bytes / (ms * 1e6));
    HIP_CHECK(hipFree(w));
    HIP_CHECK(hipFree(sink));
  }

  // ------------------------------------------------------------------
  std::printf("[2] per-shape S=1 GEMV: rocBLAS gemm_ex (engine call) vs split-K "
              "reference\n");
  double rb_floor = 0, ref_floor = 0;
  for (const auto& s : kShapes) {
    const size_t wbytes = size_t(s.N) * s.K * 2;
    unsigned short* W = (unsigned short*)dmalloc(wbytes);
    unsigned short* x = (unsigned short*)dmalloc(size_t(s.K) * 2);
    float* y = (float*)dmalloc(size_t(s.N) * 4);
    const uint32_t CK = 2048, chunks = (s.K + CK - 1) / CK;
    float* part = (float*)dmalloc(size_t(chunks) * s.N * 4);
    dbf16_fill(W, wbytes);
    dbf16_fill(x, size_t(s.K) * 2);

    // sanity: reference GEMV vs host dot product on row 0 (only small K)
    {
      std::vector<unsigned short> wh(s.K), xh(s.K);
      HIP_CHECK(hipMemcpy(wh.data(), W, s.K * 2, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(xh.data(), x, s.K * 2, hipMemcpyDeviceToHost));
      float ref = 0;
      for (uint32_t k = 0; k < s.K; ++k) {
        const auto b2f = [](unsigned short b) {
          uint32_t u = uint32_t(b) << 16;
          float f;
          std::memcpy(&f, &u, 4);
          return f;
        };
        ref += b2f(wh[k]) * b2f(xh[k]);
      }
      decode_roofline::gemv_splitk(W, x, y, part, s.N, s.K, CK);
      float y0 = 0;
      HIP_CHECK(hipMemcpy(&y0, y, 4, hipMemcpyDeviceToHost));
      std::printf("%s  row0: gpu %9.4f host %9.4f %s\n", s.name, y0, ref,
                  std::fabs(y0 - ref) < 1e-2 * std::fabs(ref) + 1e-3f ? "ok" : "MISMATCH");
    }

    const int iters = s.N > 60000 ? 30 : 100;
    const double ms_rb = bench_ms([&] { gemm_s1(handle, W, x, y, s.N, s.K); }, 10, iters);
    const double ms_wide =
        bench_ms([&] { gemm_s1_wide(handle, W, x, y, s.N, s.K); }, 10, iters);
    const double ms_ref =
        bench_ms([&] { decode_roofline::gemv_splitk(W, x, y, part, s.N, s.K, CK); }, 10,
                 iters);
    const double gb = wbytes / 1e9;
    std::printf("    rocBLAS TN: %6.3f ms = %5.0f GB/s | wide-M1: %6.3f ms = %5.0f GB/s "
                "| split-K ref: %6.3f ms = %5.0f GB/s  (%.0f MB x%d/token)\n",
                ms_rb, gb / (ms_rb / 1e3), ms_wide, gb / (ms_wide / 1e3), ms_ref,
                gb / (ms_ref / 1e3), gb, s.per_token);
    rb_floor += ms_rb * s.per_token;
    ref_floor += ms_ref * s.per_token;
    HIP_CHECK(hipFree(W));
    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(y));
    HIP_CHECK(hipFree(part));
  }
  std::printf("    GEMM-only floor/token: rocBLAS %6.2f ms (%5.1f tok/s) | split-K ref "
              "%6.2f ms (%5.1f tok/s)\n\n",
              rb_floor, 1e3 / rb_floor, ref_floor, 1e3 / ref_floor);

  // ------------------------------------------------------------------
  // Build the replica state.
  Replica r{};
  r.w_qkv = (unsigned short*)dmalloc(size_t(kQkvN) * kD * 2);
  r.w_o = (unsigned short*)dmalloc(size_t(kD) * kQd * 2);
  r.w_gup = (unsigned short*)dmalloc(size_t(2 * kInter) * kD * 2);
  r.w_down = (unsigned short*)dmalloc(size_t(kD) * kInter * 2);
  r.w_lm = (unsigned short*)dmalloc(size_t(kV) * kD * 2);
  r.residual = (float*)dmalloc(kD * 4);
  r.io = (float*)dmalloc(kD * 4);
  r.gamma = (float*)dmalloc(kD * 4);
  r.dx = (unsigned short*)dmalloc(kD * 2);
  r.dattn = (unsigned short*)dmalloc(kQd * 2);
  r.dact = (unsigned short*)dmalloc(kInter * 2);
  r.dqkv = (float*)dmalloc(kQkvN * 4);
  r.dgu = (float*)dmalloc(size_t(2 * kInter) * 4);
  r.logits = (float*)dmalloc(kV * 4);
  r.kcache = (float*)dmalloc(size_t(kMaxT) * kNkv * kHd * 4);
  r.vcache = (float*)dmalloc(size_t(kMaxT) * kNkv * kHd * 4);
  r.cos_t = (float*)dmalloc(size_t(kMaxT) * (kHd / 2) * 4);
  r.sin_t = (float*)dmalloc(size_t(kMaxT) * (kHd / 2) * 4);
  r.dsplit = (float*)dmalloc(size_t(rk::kAttnMaxSplits) * kNh * rk::kAttnSplitMaxS *
                             (kHd + 2) * 4);
  // split-K partials: largest need is lm_head (2 chunks x 128256)
  r.dpart = (float*)dmalloc(size_t(2) * kV * 4);
  r.ids_dev = (int32_t*)dmalloc(4);
  r.did = (int32_t*)dmalloc(4);
  dbf16_fill(r.w_qkv, size_t(kQkvN) * kD * 2);
  dbf16_fill(r.w_o, size_t(kD) * kQd * 2);
  dbf16_fill(r.w_gup, size_t(2 * kInter) * kD * 2);
  dbf16_fill(r.w_down, size_t(kD) * kInter * 2);
  dbf16_fill(r.w_lm, size_t(kV) * kD * 2);
  HIP_CHECK(hipMemset(r.gamma, 0x3F, kD * 4));
  HIP_CHECK(hipMemset(r.kcache, 0x3F, size_t(kMaxT) * kNkv * kHd * 4));
  HIP_CHECK(hipMemset(r.vcache, 0x3F, size_t(kMaxT) * kNkv * kHd * 4));
  HIP_CHECK(hipMemset(r.cos_t, 0x3F, size_t(kMaxT) * (kHd / 2) * 4));
  HIP_CHECK(hipMemset(r.sin_t, 0x3F, size_t(kMaxT) * (kHd / 2) * 4));
  HIP_CHECK(hipMemset(r.residual, 0, kD * 4));

  std::printf("[3] SIMT-only chain (engine kernels, exact decode sequence, S=1)\n");
  {
    const double ms = bench_ms(
        [&] { token(r, handle, 140, kSimt); }, 10, 200);
    // per-token SIMT launches: embed 1 + per layer (2 rmsnorm, 2 add, 2
    // convert, rope, attn x2, silu) + final rmsnorm + convert + argmax
    const int launches = 1 + kLayers * 10 + 3;
    std::printf("    %6.3f ms/token over %d launches (avg %.1f us/launch)\n\n", ms,
                launches, ms * 1e3 / launches);
  }

  std::printf("[4] full replica chains (exact engine kernel sequence)\n");
  {
    const double ms_async = bench_ms([&] { token(r, handle, 140, kFull); }, 5, 64);
    std::printf("    rocBLAS gemms, async:          %6.2f ms/token (%5.1f tok/s)\n",
                ms_async, 1e3 / ms_async);
  }
  {
    // engine behavior: per-token 4B D2H after argmax (the sampler's sync)
    const int N = 128;
    for (int i = 0; i < 8; ++i) token(r, handle, 140, kFull | kWithD2H);
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) token(r, handle, 140, kFull | kWithD2H);
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / N;
    std::printf("    rocBLAS gemms, per-token sync: %6.2f ms/token (%5.1f tok/s)\n", ms,
                1e3 / ms);
  }
  {
    const double ms_async =
        bench_ms([&] { token(r, handle, 140, kFull | kRefGemm); }, 5, 64);
    const int N = 128;
    for (int i = 0; i < 8; ++i) token(r, handle, 140, kFull | kRefGemm | kWithD2H);
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) token(r, handle, 140, kFull | kRefGemm | kWithD2H);
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / N;
    std::printf("    split-K ref gemms, async:      %6.2f ms/token (%5.1f tok/s)\n",
                ms_async, 1e3 / ms_async);
    std::printf("    split-K ref gemms, per-tok sy: %6.2f ms/token (%5.1f tok/s)\n", ms,
                1e3 / ms);
  }
  {
    // CPU enqueue rate: time to ISSUE one token's work, measured by issuing
    // a single token right after a sync (queue starts empty, never fills:
    // 450/810 launches < queue depth). Min over reps = pure CPU cost.
    for (auto [label, mode, nlaunch] :
         std::initializer_list<std::tuple<const char*, unsigned, int>>{
             {"rocBLAS gemms", kFull, 452},
             {"split-K ref  ", kFull | kRefGemm, 581},
             {"gemm-only rb ", kGemm, 129}}) {
      double best = 1e9;
      for (int rep = 0; rep < 20; ++rep) {
        HIP_CHECK(hipDeviceSynchronize());
        const auto t0 = Clock::now();
        token(r, handle, 140, mode);
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        HIP_CHECK(hipDeviceSynchronize());
        if (ms < best) best = ms;
      }
      std::printf("    CPU enqueue, %s: %6.2f ms/token (%d launches)\n", label, best,
                  nlaunch);
    }
  }

  std::printf("[5] phase breakdown (hipEvents, per-token sync, 32 tokens at pos 140)\n");
  {
    PhaseTimers t;
    t.ev_attn_a.resize(kLayers);
    t.ev_attn_b.resize(kLayers);
    t.ev_ffn_a.resize(kLayers);
    t.ev_ffn_b.resize(kLayers);
    for (uint32_t l = 0; l < kLayers; ++l) {
      HIP_CHECK(hipEventCreate(&t.ev_attn_a[l]));
      HIP_CHECK(hipEventCreate(&t.ev_attn_b[l]));
      HIP_CHECK(hipEventCreate(&t.ev_ffn_a[l]));
      HIP_CHECK(hipEventCreate(&t.ev_ffn_b[l]));
    }
    HIP_CHECK(hipEventCreate(&t.tail_a));
    HIP_CHECK(hipEventCreate(&t.tail_b));
    HIP_CHECK(hipDeviceSynchronize());
    const auto t0 = Clock::now();
    const int N = 32;
    double attn = 0, ffn = 0, tail = 0;
    for (int i = 0; i < N; ++i) {
      token_phased(r, handle, 140, t);
      HIP_CHECK(hipDeviceSynchronize());  // read events before re-recording
      float a;
      for (uint32_t l = 0; l < kLayers; ++l) {
        HIP_CHECK(hipEventElapsedTime(&a, t.ev_attn_a[l], t.ev_attn_b[l]));
        attn += a;
        HIP_CHECK(hipEventElapsedTime(&a, t.ev_ffn_a[l], t.ev_ffn_b[l]));
        ffn += a;
      }
      HIP_CHECK(hipEventElapsedTime(&a, t.tail_a, t.tail_b));
      tail += a;
    }
    const double wall =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / N;
    std::printf("    attn phases: %6.2f ms | ffn phases: %6.2f ms | lm+argmax: %5.2f ms "
                "| sum %6.2f | wall %6.2f (outside-gap %5.2f)\n",
                attn / N, ffn / N, tail / N, (attn + ffn + tail) / N, wall,
                wall - (attn + ffn + tail) / N);
  }

  RB_CHECK(rocblas_destroy_handle(handle));
  std::printf("\ndone\n");
  return 0;
}
