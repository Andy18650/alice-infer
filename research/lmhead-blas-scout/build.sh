#!/bin/sh
# Standalone build (research harness, deliberately outside the engine CMake;
# same style as the other research studies). The benches link the engine's
# static library for the cpu_generic::matmul_wt oracle.
#
# Usage:  ./build.sh [path/to/libopenblas.so]   # omit to skip the CPU bench
# Env:    ALICE_LIB  (default: <root>/build/libalice.a)
#         CXX        (default: g++)
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
LIB=${ALICE_LIB:-$ROOT/build/libalice.a}
CXX=${CXX:-g++}
[ -f "$LIB" ] || { echo "libalice.a not found at $LIB (build the engine first, or set ALICE_LIB)"; exit 1; }

if [ "$1" ]; then
  $CXX -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -I"$ROOT/include" bench_lmhead.cpp "$LIB" -fopenmp \
      "$1" -lamdhip64 -lrocblas -o bench_lmhead
  echo "built ./bench_lmhead (CPU OpenBLAS scout)"
else
  echo "no OpenBLAS .so given; skipping the CPU bench"
fi

$CXX -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -I"$ROOT/include" bench_lmhead_gpu.cpp \
    "$LIB" -fopenmp -lrocblas -lamdhip64 -o bench_lmhead_gpu
echo "built ./bench_lmhead_gpu"
