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
  src/blocks/cpu_generic/kernels.cpp
  src/blocks/cpu_generic/dense_ffn.cpp
  src/blocks/cpu_generic/gqa_attention.cpp
  src/sampler.cpp
  src/frontend/tokenizer.cpp
)
add_library(alice::alice ALIAS alice)
target_include_directories(alice PUBLIC include)
target_link_libraries(alice PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(alice PRIVATE -Wall -Wextra -Wpedantic)

# PCRE2 for the frontend tokenizer (pre-tokenizer regex, UTF + UCP): system
# package via pkg-config, required at configure time. No vendored copy, no
# runtime dlopen — same standard-approach rule as rocBLAS/HIP.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
target_link_libraries(alice PRIVATE PkgConfig::PCRE2)

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

# ROCm GPU (mandatory, design §3: the engine requires a GPU). Platform
# plumbing + GPU-resident model (device stream, SIMT kernels via hipcc,
# device blocks, device trunk/model).
find_library(ALICE_ROCBLAS_LIB rocblas)
find_library(ALICE_HIP_LIB amdhip64)
find_program(ALICE_HIPCC hipcc)
set(ALICE_GPU_ARCH "gfx90a" CACHE STRING "--offload-arch for the SIMT kernel TU")
if(ALICE_ROCBLAS_LIB AND ALICE_HIP_LIB AND ALICE_HIPCC AND
   EXISTS "/usr/include/rocblas/rocblas.h")
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/alice_gpu_kernels.o
      COMMAND ${ALICE_HIPCC} -O3 -std=c++17 --offload-arch=${ALICE_GPU_ARCH}
              -c ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels.hip
              -o ${CMAKE_CURRENT_BINARY_DIR}/alice_gpu_kernels.o
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels.hip
              ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels.hpp)
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu_rocm_kernels.o
      COMMAND ${ALICE_HIPCC} -O3 -std=c++17 --offload-arch=${ALICE_GPU_ARCH}
              -c ${CMAKE_CURRENT_SOURCE_DIR}/src/blocks/gpu_rocm/kernels.hip
              -o ${CMAKE_CURRENT_BINARY_DIR}/gpu_rocm_kernels.o
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/blocks/gpu_rocm/kernels.hip
              ${CMAKE_CURRENT_SOURCE_DIR}/src/blocks/gpu_rocm/kernels.hpp)
  target_sources(alice PRIVATE
      src/blocks/gpu_rocm/platform.cpp
      src/blocks/gpu_rocm/dense_ffn.cpp src/blocks/gpu_rocm/gqa_attention.cpp
      src/gpu/trunk.cpp src/gpu/embedder.cpp src/gpu/lm_head.cpp
      src/gpu/sampler.cpp src/gpu/cpu_reference.cpp src/gpu/model.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/alice_gpu_kernels.o
      ${CMAKE_CURRENT_BINARY_DIR}/gpu_rocm_kernels.o)
  # __HIP_PLATFORM_AMD__ + the HIP runtime are PUBLIC: alice/tensor.hpp (a
  # public header) allocates pinned host storage via hipHostMalloc, so every
  # consumer of the library compiles and links against HIP (user decision
  # 2026-09: the engine is GPU-mandatory; pinning in the type enforces §9 by
  # construction).
  target_compile_definitions(alice PRIVATE ALICE_GPU_RESIDENT)
  target_compile_definitions(alice PUBLIC __HIP_PLATFORM_AMD__)
  target_link_libraries(alice PRIVATE "${ALICE_ROCBLAS_LIB}")
  target_link_libraries(alice PUBLIC "${ALICE_HIP_LIB}")
  set(ALICE_GPU_RESIDENT TRUE)
  message(STATUS "alice: ROCm + hipcc found, GPU-resident engine enabled (${ALICE_GPU_ARCH})")
else()
  message(FATAL_ERROR "alice: ROCm + hipcc required (the engine is GPU-mandatory, design §3)")
endif()

# Minimal REPL PoC (design §12: the CLI owns the whole loop for now)
add_executable(alice_repl src/frontend/repl.cpp)
target_link_libraries(alice_repl PRIVATE alice)

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
    tests/test_tokenizer.cpp
  )
  if(ALICE_GPU_RESIDENT)
    target_sources(alice_tests PRIVATE tests/test_gpu_kernels.cpp
                                       tests/test_gpu_blocks.cpp
                                       tests/test_gpu_model.cpp)
    target_compile_definitions(alice_tests PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(alice_tests PRIVATE "${ALICE_HIP_LIB}")
  endif()
  target_link_libraries(alice_tests PRIVATE alice GTest::gtest_main nlohmann_json::nlohmann_json)
  target_compile_definitions(alice_tests PRIVATE ALICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(GoogleTest)
  # Easy win from the thread sweep (tools/omp_sweep.sh, docs §10): on this
  # box decode peaks at one thread per physical core (48), while all 96 is
  # slower and unstable. Set here so every ctest run gets it automatically;
  # override per-run with CTEST_ENVIRONMENT or by editing this line.
  gtest_discover_tests(alice_tests PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=48")
endif()
