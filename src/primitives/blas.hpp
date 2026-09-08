#pragma once

// Thin internal wrapper over whichever BLAS the build selected.
// We declare the handful of entry points ourselves (extern "C") instead of
// including any vendor cblas.h: the entry points we care about most
// (bf16 GEMM with fp32 accumulate, threading control) are non-standard and
// differently named across OpenBLAS / MKL / BLIS / Accelerate.

#include <cstddef>
#include <cstdint>

namespace alice::blas {

const char* backend_name();  // "none" | "openblas" | "mkl" | "accelerate"
bool available();

void set_num_threads(int n);

// C[M,N] = alpha * A[M,K] @ B[N,K]^T + beta * C   (all row-major)
void gemm_f32_nt(size_t m, size_t n, size_t k, float alpha,
                 const float* a, size_t lda,
                 const float* b, size_t ldb,
                 float beta, float* c, size_t ldc);

// bf16 inputs, fp32 accumulate and output. Not provided by every backend;
// check gemm_bf16_available() (primitives fall back to their own kernels).
bool gemm_bf16_available();
void gemm_bf16_nt(size_t m, size_t n, size_t k, float alpha,
                  const uint16_t* a, size_t lda,
                  const uint16_t* b, size_t ldb,
                  float beta, float* c, size_t ldc);

}  // namespace alice::blas
