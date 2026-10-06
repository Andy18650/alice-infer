cmake_minimum_required(VERSION 3.25)
project(alice-infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
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
  src/frontend/tokenizer.cpp
  src/sampler.cpp
)
add_library(alice::alice ALIAS alice)
target_include_directories(alice PUBLIC include PRIVATE src)
target_link_libraries(alice PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(alice PRIVATE -Wall -Wextra -Wpedantic)

# PCRE2 for the frontend tokenizer (pre-tokenizer regex, UTF + UCP): system
# package via pkg-config, required at configure time. No vendored copy, no
# runtime dlopen — same standard-approach rule as rocBLAS/HIP.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
target_link_libraries(alice PRIVATE PkgConfig::PCRE2)

# ROCm GPU (mandatory, design §3: the engine requires a GPU). Platform
# plumbing + GPU-resident model (device stream, SIMT kernels, device
# blocks, device trunk/model). The .hip TUs are first-class HIP sources
# (CMake >= 3.21 language support): compiled by hipcc, offload arch from
# ALICE_GPU_ARCH, header dependencies tracked by the build system — the
# former add_custom_command invocation is retired.
find_library(ALICE_BLAKE3_LIB blake3)
find_library(ALICE_ROCBLAS_LIB rocblas)
find_library(ALICE_HIP_LIB amdhip64)
find_library(ALICE_RCCL_LIB rccl)
if(NOT ALICE_ROCBLAS_LIB OR NOT ALICE_HIP_LIB OR NOT ALICE_BLAKE3_LIB OR
   NOT EXISTS "/usr/include/rocblas/rocblas.h" OR
   NOT EXISTS "/usr/include/blake3.h")
  message(FATAL_ERROR "alice: ROCm (rocBLAS + HIP runtime) required — the engine is GPU-mandatory (design §3)")
endif()
if(NOT ALICE_RCCL_LIB OR NOT EXISTS "/usr/include/rccl/rccl.h")
  message(FATAL_ERROR "alice: RCCL required for tensor-parallel collectives")
endif()
set(ALICE_GPU_ARCH "gfx90a" CACHE STRING "HIP offload architecture for the device TUs")
set(CMAKE_HIP_ARCHITECTURES "${ALICE_GPU_ARCH}")
# Device TUs stay optimized in every configuration: nothing source-debugs
# the kernels, and -O0 device code only breaks things — it doubles the
# Debug test suite (measured 92.8 s -> 176.9 s) and even defeats
# __launch_bounds__ occupancy targets (-Wpass-failed at -O0). The old
# hipcc custom command always passed -O3; this keeps that behavior
# explicit. Debug info still comes along via the per-config -g.
string(APPEND CMAKE_HIP_FLAGS " -O3")
enable_language(HIP)  # fails loudly at configure time without hipcc
# The target predates the language (created above, before enable_language):
# CMAKE_HIP_ARCHITECTURES only initializes targets created AFTER the
# language is enabled, so set the property explicitly.
set_target_properties(alice PROPERTIES HIP_ARCHITECTURES "${ALICE_GPU_ARCH}")

target_sources(alice PRIVATE
    src/blocks/gpu_rocm/dense_ffn.cpp src/blocks/gpu_rocm/gqa_attention.cpp
    src/blocks/dense_ffn_factory.cpp src/blocks/gqa_attention_factory.cpp
    src/blocks/block_factory.cpp
    src/data_loader.cpp
    src/gpu/resources.cpp src/gpu/trunk.cpp src/gpu/embedder.cpp src/gpu/lm_head.cpp
    src/gpu/model.cpp
    src/gpu/kernels.hip
    src/blocks/gpu_rocm/kernels.hip
    src/engine.cpp)
target_compile_definitions(alice PUBLIC JSON_DIAGNOSTICS=1)  # json exceptions carry the JSON pointer path
target_compile_definitions(alice PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(alice PRIVATE "${ALICE_ROCBLAS_LIB}" "${ALICE_HIP_LIB}" "${ALICE_BLAKE3_LIB}" "${ALICE_RCCL_LIB}")
message(STATUS "alice: HIP language enabled (${CMAKE_HIP_COMPILER}, arch ${CMAKE_HIP_ARCHITECTURES})")

# Minimal REPL PoC (design §12: the CLI owns the whole loop for now)
add_executable(alice_repl src/frontend/repl.cpp)
# The model-format path (init_engine -> Model::load_from_format) builds blocks
# via the self-registering BlockFactory registry; without --whole-archive the
# linker drops the factory TUs (no direct caller) and the registry comes up
# empty. Wrap libalice.a so the static initializers survive.
target_link_libraries(alice_repl PRIVATE
  -Wl,--whole-archive alice -Wl,--no-whole-archive nlohmann_json::nlohmann_json)

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
    tests/test_tokenizer.cpp
  )
  target_sources(alice_tests PRIVATE tests/test_gpu_kernels.cpp
                                     tests/test_gpu_blocks.cpp
                                     tests/test_gpu_model.cpp
                                     tests/test_hf_block_oracle.cpp
                                     tests/test_data_loader.cpp
                                     tests/oracle/math.cpp)
  target_include_directories(alice_tests PRIVATE tests)
  target_compile_definitions(alice_tests PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(alice_tests PRIVATE "${ALICE_HIP_LIB}")
  # --whole-archive: the model-format path (init_engine, used by test_gpu_model)
  # builds blocks via the self-registering BlockFactory registry; without it
  # the factory TUs are dropped and the registry comes up empty.
  target_link_libraries(alice_tests PRIVATE
    -Wl,--whole-archive alice -Wl,--no-whole-archive GTest::gtest_main nlohmann_json::nlohmann_json)
  target_compile_definitions(alice_tests PRIVATE ALICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(GoogleTest)
  gtest_discover_tests(alice_tests)
endif()
