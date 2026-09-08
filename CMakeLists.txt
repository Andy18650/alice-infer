cmake_minimum_required(VERSION 3.25)
project(alice-infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALICE_BUILD_TESTS "Build unit tests" ON)
set(ALICE_BLAS "AUTO" CACHE STRING "BLAS backend: AUTO | NONE | OPENBLAS | MKL | ACCELERATE")
set_property(CACHE ALICE_BLAS PROPERTY STRINGS AUTO NONE OPENBLAS MKL ACCELERATE)

# --- JSON (private dependency: used in .cpp files only, never in public API) ---
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(alice
  src/arena.cpp
  src/safetensors.cpp
  src/manifest.cpp
  src/block.cpp
  src/model.cpp
  src/primitives/stubs.cpp
  src/primitives/blas.cpp
)
add_library(alice::alice ALIAS alice)

target_include_directories(alice PUBLIC include)
target_link_libraries(alice PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(alice PRIVATE -Wall -Wextra -Wpedantic)

# --- BLAS detection (compile-time backend selection; thin wrapper in src/primitives/blas.cpp) ---
set(ALICE_BLAS_BACKEND NONE)
if(NOT ALICE_BLAS STREQUAL "NONE")
  if(ALICE_BLAS STREQUAL "OPENBLAS" OR ALICE_BLAS STREQUAL "AUTO")
    find_library(ALICE_OPENBLAS_LIB NAMES openblas)
    if(ALICE_OPENBLAS_LIB)
      set(ALICE_BLAS_BACKEND OPENBLAS)
      target_link_libraries(alice PRIVATE ${ALICE_OPENBLAS_LIB})
    elseif(ALICE_BLAS STREQUAL "OPENBLAS")
      message(FATAL_ERROR "ALICE_BLAS=OPENBLAS but libopenblas was not found")
    endif()
  endif()
  if(ALICE_BLAS_BACKEND STREQUAL "NONE" AND (ALICE_BLAS STREQUAL "MKL" OR ALICE_BLAS STREQUAL "AUTO"))
    find_package(MKL QUIET)
    if(MKL_FOUND)
      set(ALICE_BLAS_BACKEND MKL)
      target_link_libraries(alice PRIVATE MKL::MKL)
    elseif(ALICE_BLAS STREQUAL "MKL")
      message(FATAL_ERROR "ALICE_BLAS=MKL but MKL was not found")
    endif()
  endif()
  if(APPLE AND (ALICE_BLAS STREQUAL "ACCELERATE" OR
               (ALICE_BLAS STREQUAL "AUTO" AND ALICE_BLAS_BACKEND STREQUAL "NONE")))
    find_library(ALICE_ACCELERATE_FW Accelerate)
    if(ALICE_ACCELERATE_FW)
      set(ALICE_BLAS_BACKEND ACCELERATE)
      target_link_libraries(alice PRIVATE ${ALICE_ACCELERATE_FW})
    endif()
  endif()
endif()
message(STATUS "alice: BLAS backend = ${ALICE_BLAS_BACKEND}")
target_compile_definitions(alice PRIVATE ALICE_BLAS_BACKEND_${ALICE_BLAS_BACKEND})

# --- Tests ---
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
  )
  target_link_libraries(alice_tests PRIVATE alice GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(alice_tests)
endif()
