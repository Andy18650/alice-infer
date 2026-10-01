#pragma once

// BLAS backend selection (PROTOTYPE 2026-09: rocBLAS only for now). The
// handle type is shim-able — a future CUDA/MUSA backend adds an #elif branch
// defining BlasHandle to its own handle, and the Resources interface (and
// anything that stores a BlasHandle) is unchanged. Conditional compile: the
// rocBLAS path is compiled when __HIP_PLATFORM_AMD__ is defined (the alice
// target defines it PRIVATE). Per the user direction: BLAS L1/2/3 calls are
// shim-able with macros + conditional compile; this header is the seed (the
// handle type). Full call-site shimming (gemm etc.) is deferred.

#if defined(__HIP_PLATFORM_AMD__)
#include <rocblas/rocblas.h>
namespace alice::gpu {
using BlasHandle = rocblas_handle;
}
#else
#error "unsupported BLAS backend: only __HIP_PLATFORM_AMD__ (rocBLAS) is built for now"
#endif
