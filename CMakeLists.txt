cmake_minimum_required(VERSION 3.25)
project(alice-infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALICE_BUILD_TESTS "Build unit tests" ON)

# --- JSON (private dependency: used in .cpp files only, never in public API) ---
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# The engine is being rebuilt from docs/engine-design.md.
add_library(alice
  src/safetensors.cpp
  src/manifest.cpp
  src/hardware.cpp
  src/blocks/dense_ffn_bf16.cpp
  src/blocks/gqa_attention_bf16.cpp
  src/blocks/cpu_generic/kernels.cpp
  src/blocks/cpu_generic/dense_ffn.cpp
  src/blocks/cpu_generic/gqa_attention.cpp
  src/blocks/cpu_avx2/dense_ffn.cpp
  src/embedder.cpp
  src/lm_head.cpp
  src/sampler.cpp
  src/trunk.cpp
  src/model.cpp
  src/frontend/tokenizer.cpp
  src/frontend/pcre2.cpp
)
add_library(alice::alice ALIAS alice)
target_include_directories(alice PUBLIC include)
target_link_libraries(alice PRIVATE nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
target_compile_options(alice PRIVATE -Wall -Wextra -Wpedantic)

# The CPU_AVX2 operator TU gets ISA flags per-file; the rest of the binary
# stays portable. Dispatch is runtime-guarded (cpuid) in the factory.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set_source_files_properties(src/blocks/cpu_avx2/dense_ffn.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# OpenMP for intra-op kernel parallelism (design §10: each implementation's
# private business). Sequential fallback when unavailable (pragma macro
# compiles to nothing); OMP_NUM_THREADS=1 reproduces the baseline.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(alice PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "alice: OpenMP enabled (${OpenMP_CXX_VERSION})")
else()
  message(STATUS "alice: OpenMP not found, kernels run sequentially")
endif()

# ROCm GPU (GPU_ROCBLAS hardware class; auto-selected when present, interim
# until the loader overhaul). Host-only C++ against the C APIs — no hipcc;
# custom device kernels compile at runtime via hipRTC. When absent, the
# factories' fallback chains simply resolve to the CPU implementations.
find_library(ALICE_ROCBLAS_LIB rocblas)
find_library(ALICE_HIP_LIB amdhip64)
find_library(ALICE_HIPRTC_LIB hiprtc)
if(ALICE_ROCBLAS_LIB AND ALICE_HIP_LIB AND ALICE_HIPRTC_LIB AND
   EXISTS "/usr/include/rocblas/rocblas.h")
  target_sources(alice PRIVATE src/blocks/gpu_rocblas/platform.cpp
                               src/blocks/gpu_rocblas/dense_ffn.cpp)
  target_compile_definitions(alice PRIVATE ALICE_HAVE_ROCBLAS __HIP_PLATFORM_AMD__)
  target_link_libraries(alice PRIVATE "${ALICE_ROCBLAS_LIB}" "${ALICE_HIP_LIB}"
                                      "${ALICE_HIPRTC_LIB}")
  set(ALICE_HAVE_ROCBLAS TRUE)
  message(STATUS "alice: ROCm found, GPU_ROCBLAS hardware class enabled")
else()
  message(STATUS "alice: ROCm not found, GPU_ROCBLAS hardware class disabled")
endif()

# Minimal REPL PoC (design §12: the CLI owns the whole loop for now)
add_executable(alice_repl src/frontend/repl.cpp)
target_link_libraries(alice_repl PRIVATE alice)

# Per-block microbenchmark (dense FFN; pattern for future per-operator benches)
add_executable(alice_bench_ffn src/frontend/bench_ffn.cpp)
target_link_libraries(alice_bench_ffn PRIVATE alice)

# BLAS scouting bench for lm_head (concluded experiment, kept as evidence;
# builds only when ALICE_OPENBLAS_SO points at an OpenBLAS .so, e.g. the
# system libopenblas0 package. No BLAS dependency in alice itself.)
set(ALICE_OPENBLAS_SO "" CACHE FILEPATH
    "path to an OpenBLAS shared object for the scout bench")
if(EXISTS "${ALICE_OPENBLAS_SO}")
  add_executable(alice_bench_lmhead src/frontend/bench_lmhead.cpp)
  target_link_libraries(alice_bench_lmhead PRIVATE alice "${ALICE_OPENBLAS_SO}")
else()
  message(STATUS "OpenBLAS .so not found at ALICE_OPENBLAS_SO; skipping bench_lmhead")
endif()

# rocBLAS scouting bench for lm_head (GPU counterpart; host-only C++ against
# the C APIs — no hipcc, Tensile code objects are pre-compiled).
if(ALICE_HAVE_ROCBLAS)
  add_executable(alice_bench_lmhead_gpu src/frontend/bench_lmhead_gpu.cpp)
  # hipcc defines this implicitly; plain g++ must pick a platform explicitly.
  target_compile_definitions(alice_bench_lmhead_gpu PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(alice_bench_lmhead_gpu PRIVATE alice "${ALICE_ROCBLAS_LIB}"
                                                       "${ALICE_HIP_LIB}")
else()
  message(STATUS "rocBLAS/HIP not found; skipping bench_lmhead_gpu")
endif()

if(ALICE_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_executable(alice_tests
    tests/test_tensor.cpp
    tests/test_safetensors.cpp
    tests/test_manifest.cpp
    tests/test_blocks.cpp
    tests/test_model.cpp
    tests/test_tokenizer.cpp
    tests/test_canonical.cpp
  )
  target_link_libraries(alice_tests PRIVATE alice GTest::gtest_main nlohmann_json::nlohmann_json)
  target_compile_definitions(alice_tests PRIVATE ALICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(GoogleTest)
  # Easy win from the thread sweep (tools/omp_sweep.sh, docs §10): on this
  # box decode peaks at one thread per physical core (48), while all 96 is
  # slower and unstable. Set here so every ctest run gets it automatically;
  # override per-run with CTEST_ENVIRONMENT or by editing this line.
  gtest_discover_tests(alice_tests PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=48")
endif()
