#!/bin/sh
# Self-contained research build. Never builds or links the production engine.
set -eu
cd "$(dirname "$0")"
root=$(CDPATH= cd ../.. && pwd)
out="$root/temp/p2p-allreduce"
mkdir -p "$out"
hipcc -O3 -std=c++20 --offload-arch=gfx90a -c kernels.hip -o "$out/kernels.o"
g++ -O3 -std=c++20 -Wall -Wextra -Wpedantic -D__HIP_PLATFORM_AMD__ \
  study.cpp "$out/kernels.o" -lamdhip64 -o "$out/study"
echo "built $out/study"
