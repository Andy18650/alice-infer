#!/bin/sh
# Standalone build (research harness, deliberately outside engine CMake; same
# style as the other research studies). kernels.hip via hipcc, study.cpp via
# g++ linking the HIP runtime (+ RCCL for the baseline backend).
set -e
cd "$(dirname "$0")"

# kernels.hip: hipcc-compiled device kernels (fill, sum_reduce).
hipcc -O3 -std=c++17 --offload-arch=gfx90a -c kernels.hip -o kernels.hip.o

# study.cpp: g++ host orchestration. __HIP_PLATFORM_AMD__ unlocks the HIP
# host API headers; USE_RCCL pulls in rccl.h and -lrccl (the rccl backend needs
# the AMD-overlay librccl on LD_LIBRARY_PATH at run time; Canonical's build
# segfaults on >=2 ranks).
g++ -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -DUSE_RCCL \
    -I. study.cpp kernels.hip.o \
    -lamdhip64 -lrccl -o study

# Device-flag barrier / allreduce variants (atomics.md). Needs -lrccl only
# for the in-harness RCCL reference phase; run with the AMD-overlay librccl
# on LD_LIBRARY_PATH.
hipcc -O2 --offload-arch=gfx90a -I/usr/include/rccl flagprobe.hip -o flagprobe -lrccl -pthread

# Coherency/microarchitecture probes (atomics.md, README §Coherency probes).
for p in l2probe bwprobe clockprobe kbprobe atomicprobe l2alloc l2pool; do
  hipcc -O2 --offload-arch=gfx90a $p.hip -o $p -pthread
done
# fence_sass.hip is a SASS-dump source (no main): compile only.
hipcc -O2 --offload-arch=gfx90a -c fence_sass.hip -o /dev/null

# hipDeviceEnablePeerAccess bisect series (README finding #4).
for p in enableprobe enableprobe2 enableprobe3 enableprobe4; do
  g++ -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ $p.cpp -o $p -lamdhip64 -lrocblas -pthread
done

echo "built ./study ./flagprobe + probes"
