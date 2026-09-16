// BLAS scouting bench for the lm_head GEMM (user-directed experiment):
// can an off-the-shelf BLAS beat hand-rolled kernels here, and if so, is a
// BLAS-backed FFN impl worth writing?
//
// Available BLAS reality (2026-07, this machine): Ubuntu OpenBLAS 0.3.32
// pthread/LP64, FP32 cblas_sgemm ONLY — no cblas_sbgemm in any obtainable
// build (scipy wheel and Debian both disable bf16; a source build's sbgemm
// is a convert-to-fp32 fallback on Zen3 anyway). So the BLAS paths pay a
// bf16->fp32 weight conversion, measured two ways:
//   * convert:  conversion INSIDE the timed call (the honest per-call cost)
//   * cached:   weights pre-converted once (amortization ceiling; 2x memory)
// The oracle is the production path: cpu_generic::matmul_wt on bf16 weights
// (no conversion, half the weight traffic).
//
// Shapes: lm_head of the 1B — W [V=128256, d=2048] bf16, x [S, d] fp32,
// C [S, V] fp32. Reports ms/call, GFLOP/s, conversion cost, max|Delta| vs
// oracle.
//
// Usage: alice_bench_lmhead [iters_s1] [iters_big]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "alice/tensor.hpp"

// Internal kernel, self-declared (same style as the BLAS prototypes below;
// keeps the bench free of src/ include paths).
namespace alice::cpu_generic {
void matmul_wt(F32View in, BF16View w, F32View out);
}

#if defined(_OPENMP)
#include <omp.h>
#endif

// Self-declared CBLAS prototypes (design §10: no BLAS headers; Ubuntu
// OpenBLAS LP64 => int). Linked against the extracted .so by absolute path.
extern "C" {
void cblas_sgemm(int layout, int ta, int tb, int m, int n, int k, float alpha,
                 const float* A, int lda, const float* B, int ldb, float beta, float* C,
                 int ldc);
void openblas_set_num_threads(int n);
}
constexpr int kRowMajor = 101, kNoTrans = 111, kTrans = 112;

namespace {

using namespace alice;
using Clock = std::chrono::steady_clock;

constexpr uint32_t V = 128256, D = 2048;  // 1B lm_head

float pattern(size_t i) {  // bf16-exact values in [-1, 1)
  return (int((i * 2654435761ull) % 1024) - 512) / 512.0f;
}

double now_s() { return Clock::now().time_since_epoch().count() / 1e9; }

struct Result {
  double ms = 0, convert_ms = 0;
  float max_delta = 0;
};

void convert_w(const bf16* w, float* wf) {
#pragma omp parallel for
  for (size_t i = 0; i < size_t(V) * D; ++i) wf[i] = w[i].to_f32();
}

Result run(int S, const float* x, const bf16* w, const float* wf_cached,
           float* wf_scratch, float* c_blas, const float* c_ref, int iters, bool cached) {
  Result r;
  const double t0 = now_s();
  for (int it = 0; it < iters; ++it) {
    if (cached) {
      cblas_sgemm(kRowMajor, kNoTrans, kTrans, S, V, D, 1.0f, x, D, wf_cached, D, 0.0f,
                  c_blas, V);
    } else {
      const double tc = now_s();
      convert_w(w, wf_scratch);
      r.convert_ms += (now_s() - tc) * 1e3;
      cblas_sgemm(kRowMajor, kNoTrans, kTrans, S, V, D, 1.0f, x, D, wf_scratch, D, 0.0f,
                  c_blas, V);
    }
  }
  r.ms = (now_s() - t0) * 1e3 / iters;
  r.convert_ms /= iters;
  for (size_t i = 0; i < size_t(S) * V; ++i) {
    const float d = std::fabs(c_blas[i] - c_ref[i]);
    if (d > r.max_delta) r.max_delta = d;
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const int it_s1 = argc > 1 ? std::atoi(argv[1]) : 20;
  const int it_big = argc > 2 ? std::atoi(argv[2]) : 5;
#if defined(_OPENMP)
  openblas_set_num_threads(omp_get_max_threads());
#endif

  std::unique_ptr<bf16[]> w(new bf16[size_t(V) * D]);
  std::unique_ptr<float[]> wf(new float[size_t(V) * D]);
  std::unique_ptr<float[]> wf2(new float[size_t(V) * D]);  // convert-variant scratch
  std::unique_ptr<float[]> x(new float[size_t(128) * D]);
  std::unique_ptr<float[]> c_ref(new float[size_t(128) * V]);
  std::unique_ptr<float[]> c_blas(new float[size_t(128) * V]);
#pragma omp parallel for
  for (size_t i = 0; i < size_t(V) * D; ++i) w[i] = bf16::from_f32(pattern(i));
#pragma omp parallel for
  for (size_t i = 0; i < size_t(128) * D; ++i) x[i] = pattern(i ^ 0x5bd1e995u);
  convert_w(w.get(), wf.get());  // the cached variant's one-time cost (untimed)

  const double gflop = 2.0 * V * D / 1e9;  // per S=1
  std::printf("lm_head BLAS scout: V=%u d=%u, W bf16 %.0f MB (fp32 %.0f MB)\n", V, D,
              size_t(V) * D * 2 / 1e6, size_t(V) * D * 4 / 1e6);
  std::printf("%-24s %4s %9s %9s %9s %10s\n", "path", "S", "ms/call", "conv_ms", "GFLOP/s",
              "max|Delta|");

  for (const int S : {1, 8, 128}) {
    const int iters = S == 1 ? it_s1 : it_big;
    // oracle: production path, bf16 weights, no conversion
    double t0 = now_s();
    for (int it = 0; it < iters; ++it)
      cpu_generic::matmul_wt(F32View{x.get(), uint32_t(S), D, 0},
                             BF16View{w.get(), V, D, 0},
                             F32View{c_ref.get(), uint32_t(S), V, 0});
    const double ms_ref = (now_s() - t0) * 1e3 / iters;
    std::printf("%-24s %4d %9.2f %9s %9.1f %10s\n", "oracle bf16 (prod)", S, ms_ref, "-",
                S * gflop / (ms_ref / 1e3), "-");

    const auto rc = run(S, x.get(), w.get(), wf.get(), wf.get(), c_blas.get(), c_ref.get(),
                        iters, true);
    std::printf("%-24s %4d %9.2f %9s %9.1f %10.5f\n", "openblas fp32 cached", S, rc.ms,
                "-", S * gflop / (rc.ms / 1e3), rc.max_delta);

    const auto rv = run(S, x.get(), w.get(), wf.get(), wf2.get(), c_blas.get(),
                        c_ref.get(), iters, false);
    std::printf("%-24s %4d %9.2f %9.2f %9.1f %10.5f\n", "openblas fp32 convert", S, rv.ms,
                rv.convert_ms, S * gflop / (rv.ms / 1e3), rv.max_delta);
  }
  return 0;
}
