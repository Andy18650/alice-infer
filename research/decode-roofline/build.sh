#!/bin/sh
# Standalone build (research harness, deliberately outside the engine CMake;
# same style as the other research studies). Links the ENGINE'S REAL kernel
# objects (CMake HIP-language per-source objects) so the replica runs the
# exact production kernels, plus a hipcc-compiled reference module
# (gemv_ref.hip: pure-read ceiling + split-K GEMV).
#
# Env: OBJDIR (default <root>/build-release)   CXX (default g++)
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
OBJDIR=${OBJDIR:-$ROOT/build-release}
CXX=${CXX:-g++}

# HIP-language object layout: CMakeFiles/<target>.dir/<source-path>.o
GPU_KO="$OBJDIR/CMakeFiles/alice.dir/src/gpu/kernels.hip.o"
ROCM_KO="$OBJDIR/CMakeFiles/alice.dir/src/blocks/gpu_rocm/kernels.hip.o"
for o in "$GPU_KO" "$ROCM_KO"; do
  [ -f "$o" ] || { echo "missing $o (build the engine first, or set OBJDIR)"; exit 1; }
done

hipcc -O3 -std=c++17 --offload-arch=gfx90a -c gemv_ref.hip -o gemv_ref.o
$CXX -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ bench.cpp gemv_ref.o \
    "$GPU_KO" "$ROCM_KO" \
    -lrocblas -lamdhip64 -o bench_decode
echo "built ./bench_decode"
