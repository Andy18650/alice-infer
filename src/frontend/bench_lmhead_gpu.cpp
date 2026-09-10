// rocBLAS scouting bench for the lm_head GEMM (user-directed experiment,
// GPU counterpart of bench_lmhead.cpp): one device, W [V, d] bf16 staged
// once, per call x [S, d] fp32 -> host-convert bf16 -> H2D -> gemm_ex
// (bf16 in, FP32 compute, fp32 out) -> D2H logits. Per-phase timing
// breakdown (convert / H2D / gemm / D2H), GFLOP/s, effective weight
// bandwidth, max|Delta| vs the CPU oracle (fp32-x matmul_wt).
//
// rocBLAS is column-major: row-major C = x.W^T  <=>  col-major C' = W.x^T,
// so A = W (col-major [d, V], trans) and B = x (col-major [d, S]).
//
// No hipcc anywhere: host-only C++ against the C APIs; Tensile ships
// pre-compiled gfx90a bf16-HPA code objects loaded at runtime.
//
// Usage: alice_bench_lmhead_gpu [iters_s1] [iters_big] [device]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "alice/tensor.hpp"

namespace alice::cpu_generic {
void matmul_wt(F32View in, BF16View w, F32View out);  // self-declared internal
}

namespace {

using namespace alice;
using Clock = std::chrono::steady_clock;

constexpr uint32_t V = 128256, D = 2048;  // 1B lm_head

#define HIP_CHECK(x)                                                     \
  do {                                                                   \
    const hipError_t e_ = (x);                                           \
    if (e_ != hipSuccess) {                                              \
      std::fprintf(stderr, "HIP error %d at %s:%d\n", int(e_), __FILE__, __LINE__); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)
#define RB_CHECK(x)                                                      \
  do {                                                                   \
    const rocblas_status s_ = (x);                                       \
    if (s_ != rocblas_status_success) {                                  \
      std::fprintf(stderr, "rocBLAS error %d at %s:%d\n", int(s_), __FILE__, __LINE__); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

float pattern(size_t i) { return (int((i * 2654435761ull) % 1024) - 512) / 512.0f; }
double now_s() { return Clock::now().time_since_epoch().count() / 1e9; }

}  // namespace

int main(int argc, char** argv) {
  const int it_s1 = argc > 1 ? std::atoi(argv[1]) : 50;
  const int it_big = argc > 2 ? std::atoi(argv[2]) : 20;
  const int dev = argc > 3 ? std::atoi(argv[3]) : 0;

  char ver[128];
  rocblas_get_version_string(ver, sizeof(ver));
  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  std::printf("rocBLAS %s | device %d: %s (%s), %.0f MB\n", ver, dev, prop.name,
              prop.gcnArchName, prop.totalGlobalMem / 1e6);

  // host data; xb/c_gpu PINNED (pageable D2H on this stack runs a ~1 GB/s
  // staged path — measured; pinned host buffers are the fix)
  std::unique_ptr<bf16[]> w(new bf16[size_t(V) * D]);
  std::unique_ptr<float[]> x(new float[size_t(128) * D]);
  std::unique_ptr<float[]> c_ref(new float[size_t(128) * V]);
  bf16* xb = nullptr;
  float* c_gpu = nullptr;
  HIP_CHECK(hipHostMalloc(&xb, size_t(128) * D * 2, hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&c_gpu, size_t(128) * V * 4, hipHostMallocDefault));
#pragma omp parallel for
  for (size_t i = 0; i < size_t(V) * D; ++i) w[i] = bf16::from_f32(pattern(i));
#pragma omp parallel for
  for (size_t i = 0; i < size_t(128) * D; ++i) x[i] = pattern(i ^ 0x5bd1e995u);

  // device buffers (staged once: the "resident weights" of a real impl)
  void *dw = nullptr, *dx = nullptr, *dc = nullptr;
  HIP_CHECK(hipMalloc(&dw, size_t(V) * D * 2));
  HIP_CHECK(hipMalloc(&dx, size_t(128) * D * 2));
  HIP_CHECK(hipMalloc(&dc, size_t(128) * V * 4));
  HIP_CHECK(hipMemcpy(dw, w.get(), size_t(V) * D * 2, hipMemcpyHostToDevice));

  rocblas_handle handle;
  RB_CHECK(rocblas_create_handle(&handle));
  const float alpha = 1.0f, beta = 0.0f;

  auto gemm = [&](int S) {
    RB_CHECK(rocblas_gemm_ex(handle, rocblas_operation_transpose,
                             rocblas_operation_none, V, S, D, &alpha, dw,
                             rocblas_datatype_bf16_r, D, dx, rocblas_datatype_bf16_r, D,
                             &beta, dc, rocblas_datatype_f32_r, V, dc,
                             rocblas_datatype_f32_r, V, rocblas_datatype_f32_r,
                             rocblas_gemm_algo_standard, 0, 0));
  };

  // warmup (Tensile code-object load + solution selection)
  for (int i = 0; i < 3; ++i) gemm(1);
  HIP_CHECK(hipDeviceSynchronize());

  const double gflop = 2.0 * V * D / 1e9;
  std::printf("%-22s %4s %8s %8s %8s %8s %8s %9s %9s %10s\n", "path", "S", "ms/call",
              "cvt", "H2D", "gemm", "D2H", "GFLOP/s", "GB/s W", "max|Delta|");

  for (const int S : {1, 8, 128}) {
    const int iters = S == 1 ? it_s1 : it_big;

    // CPU oracle reference (production path, fp32 x)
    cpu_generic::matmul_wt(F32View{x.get(), uint32_t(S), D, 0}, BF16View{w.get(), V, D, 0},
                           F32View{c_ref.get(), uint32_t(S), V, 0});

    double tc = 0, th = 0, tg = 0, td = 0;
    const double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
      double t = now_s();
      for (size_t i = 0; i < size_t(S) * D; ++i) xb[i] = bf16::from_f32(x[i]);
      tc += now_s() - t;
      t = now_s();
      HIP_CHECK(hipMemcpy(dx, xb, size_t(S) * D * 2, hipMemcpyHostToDevice));
      th += now_s() - t;
      t = now_s();
      gemm(S);  // rocBLAS 5.1 has no bf16 gemv; n=1 GEMM is the S==1 path
      HIP_CHECK(hipDeviceSynchronize());
      tg += now_s() - t;
      t = now_s();
      HIP_CHECK(hipMemcpy(c_gpu, dc, size_t(S) * V * 4, hipMemcpyDeviceToHost));
      td += now_s() - t;
    }
    const double tot = (now_s() - t0) / iters;
    float max_d = 0;
    for (size_t i = 0; i < size_t(S) * V; ++i) {
      const float d = std::fabs(c_gpu[i] - c_ref[i]);
      if (d > max_d) max_d = d;
    }
    const double ms = tot * 1e3;
    std::printf("%-22s %4d %8.3f %8.4f %8.3f %8.3f %8.3f %9.1f %9.0f %10.5f\n",
                "rocblas gemm_ex (pinned)", S, ms,
                tc / iters * 1e3, th / iters * 1e3, tg / iters * 1e3, td / iters * 1e3,
                S * gflop / tot, size_t(V) * D * 2 / tot / 1e9, max_d);
  }

  RB_CHECK(rocblas_destroy_handle(handle));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dc));
  HIP_CHECK(hipHostFree(xb));
  HIP_CHECK(hipHostFree(c_gpu));
  return 0;
}
