#include "blas.hpp"

#include <stdexcept>

// Self-declared prototypes, on purpose. Values of the CBLAS enums are part
// of the de-facto ABI shared by netlib CBLAS, OpenBLAS and MKL.
#if defined(ALICE_BLAS_BACKEND_OPENBLAS) || defined(ALICE_BLAS_BACKEND_MKL) || \
    defined(ALICE_BLAS_BACKEND_ACCELERATE)
extern "C" {
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
void cblas_sgemm(int order, int trans_a, int trans_b, int m, int n, int k, float alpha,
                 const float* a, int lda, const float* b, int ldb, float beta, float* c,
                 int ldc);
}
#endif

#if defined(ALICE_BLAS_BACKEND_OPENBLAS)
extern "C" {
void openblas_set_num_threads(int n);
// bf16 in, fp32 out — OpenBLAS extension. VERIFY the exact signature against
// the installed OpenBLAS version before enabling; this family of entry
// points is not stable across vendors (MKL: cblas_gemm_bf16bf16f32).
void cblas_sbgemm(int order, int trans_a, int trans_b, int m, int n, int k, float alpha,
                  const uint16_t* a, int lda, const uint16_t* b, int ldb, float beta,
                  float* c, int ldc);
}
#endif

#if defined(ALICE_BLAS_BACKEND_MKL)
extern "C" {
void MKL_Set_Num_Threads(int n);
// MKL bf16 entry point; different name AND different signature family than
// OpenBLAS — this divergence is why the wrapper exists.
void cblas_gemm_bf16bf16f32(int order, int trans_a, int trans_b, int m, int n, int k,
                            float alpha, const uint16_t* a, int lda, const uint16_t* b,
                            int ldb, float beta, float* c, int ldc);
}
#endif

namespace alice::blas {

namespace {
[[noreturn]] void no_blas(const char* f) {
  throw std::logic_error(std::string("alice::blas::") + f +
                         ": library was built with ALICE_BLAS=NONE");
}
}  // namespace

const char* backend_name() {
#if defined(ALICE_BLAS_BACKEND_OPENBLAS)
  return "openblas";
#elif defined(ALICE_BLAS_BACKEND_MKL)
  return "mkl";
#elif defined(ALICE_BLAS_BACKEND_ACCELERATE)
  return "accelerate";
#else
  return "none";
#endif
}

bool available() {
#if defined(ALICE_BLAS_BACKEND_NONE)
  return false;
#else
  return true;
#endif
}

void set_num_threads(int n) {
#if defined(ALICE_BLAS_BACKEND_OPENBLAS)
  openblas_set_num_threads(n);
#elif defined(ALICE_BLAS_BACKEND_MKL)
  MKL_Set_Num_Threads(n);
#else
  (void)n;
#endif
}

void gemm_f32_nt(size_t m, size_t n, size_t k, float alpha, const float* a, size_t lda,
                 const float* b, size_t ldb, float beta, float* c, size_t ldc) {
#if defined(ALICE_BLAS_BACKEND_OPENBLAS) || defined(ALICE_BLAS_BACKEND_MKL) || \
    defined(ALICE_BLAS_BACKEND_ACCELERATE)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(m), int(n), int(k), alpha, a,
              int(lda), b, int(ldb), beta, c, int(ldc));
#else
  (void)m; (void)n; (void)k; (void)alpha; (void)a; (void)lda;
  (void)b; (void)ldb; (void)beta; (void)c; (void)ldc;
  no_blas("gemm_f32_nt");
#endif
}

bool gemm_bf16_available() {
#if defined(ALICE_BLAS_BACKEND_OPENBLAS) || defined(ALICE_BLAS_BACKEND_MKL)
  return true;
#else
  return false;
#endif
}

void gemm_bf16_nt(size_t m, size_t n, size_t k, float alpha, const uint16_t* a,
                  size_t lda, const uint16_t* b, size_t ldb, float beta, float* c,
                  size_t ldc) {
#if defined(ALICE_BLAS_BACKEND_OPENBLAS)
  cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(m), int(n), int(k), alpha, a,
               int(lda), b, int(ldb), beta, c, int(ldc));
#elif defined(ALICE_BLAS_BACKEND_MKL)
  cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, int(m), int(n), int(k),
                         alpha, a, int(lda), b, int(ldb), beta, c, int(ldc));
#else
  (void)m; (void)n; (void)k; (void)alpha; (void)a; (void)lda;
  (void)b; (void)ldb; (void)beta; (void)c; (void)ldc;
  no_blas("gemm_bf16_nt");
#endif
}

}  // namespace alice::blas
