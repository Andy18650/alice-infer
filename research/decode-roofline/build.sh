#!/bin/sh
# Standalone build (research harness, deliberately outside the engine CMake;
# same style as the other research studies). Links the ENGINE'S REAL kernel
# objects (build*/alice_gpu_kernels.o + gpu_rocm_kernels.o) so the replica
# runs the exact production kernels, plus a hipcc-compiled reference module
# (gemv_ref.hip: pure-read ceiling + split-K GEMV).
#
# Env: OBJDIR (default <root>/build-release)   CXX (default g++)
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
OBJDIR=${OBJDIR:-$ROOT/build-release}
CXX=${CXX:-g++}

for o in "$OBJDIR/alice_gpu_kernels.o" "$OBJDIR/gpu_rocm_kernels.o"; do
  [ -f "$o" ] || { echo "missing $o (build the engine first, or set OBJDIR)"; exit 1; }
done

hipcc -O3 -std=c++17 --offload-arch=gfx90a -c gemv_ref.hip -o gemv_ref.o
$CXX -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ bench.cpp gemv_ref.o \
    "$OBJDIR/alice_gpu_kernels.o" "$OBJDIR/gpu_rocm_kernels.o" \
    -lrocblas -lamdhip64 -o bench_decode
echo "built ./bench_decode"
