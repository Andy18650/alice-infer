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
  src/blocks/dense_ffn_bf16.cpp
  src/blocks/gqa_attention_bf16.cpp
  src/blocks/cpu_generic/kernels.cpp
  src/blocks/cpu_generic/dense_ffn.cpp
  src/blocks/cpu_generic/gqa_attention.cpp
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
    tests/test_model.cpp
    tests/test_tokenizer.cpp
    tests/test_canonical.cpp
  )
  target_link_libraries(alice_tests PRIVATE alice GTest::gtest_main nlohmann_json::nlohmann_json)
  target_compile_definitions(alice_tests PRIVATE ALICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(GoogleTest)
  gtest_discover_tests(alice_tests)
endif()
