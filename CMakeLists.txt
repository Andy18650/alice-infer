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
)
add_library(alice::alice ALIAS alice)
target_include_directories(alice PUBLIC include)
target_link_libraries(alice PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(alice PRIVATE -Wall -Wextra -Wpedantic)

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
  target_compile_definitions(alice_tests PRIVATE ALICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(GoogleTest)
  gtest_discover_tests(alice_tests)
endif()
