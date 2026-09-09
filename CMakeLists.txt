cmake_minimum_required(VERSION 3.25)
project(alice-infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALICE_BUILD_TESTS "Build unit tests" ON)

# The engine is being rebuilt from docs/engine-design.md. Only TensorView
# (design §4) survives the purge, so the core is header-only for now;
# compiled sources (safetensors loader, manifest, blocks, primitives)
# return as the rebuild proceeds. JSON/BLAS plumbing lives in git history.
add_library(alice INTERFACE)
add_library(alice::alice ALIAS alice)
target_include_directories(alice INTERFACE include)

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
  add_executable(alice_tests tests/test_tensor.cpp)
  target_link_libraries(alice_tests PRIVATE alice GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(alice_tests)
endif()
