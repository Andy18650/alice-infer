#!/bin/sh
# Self-contained research build. Never builds or links the production engine.
set -eu
cd "$(dirname "$0")"
root=$(CDPATH= cd ../.. && pwd)
out="$root/temp/p2p-allreduce"
mkdir -p "$out"
if [ "${1:-}" = "--counter" ]; then
  hipcc -O3 -std=c++20 --offload-arch=gfx90a counter.hip -o "$out/counter"
  echo "built $out/counter"
  exit 0
fi
hipcc -O3 -std=c++20 --offload-arch=gfx90a -c kernels.hip -o "$out/kernels.o"
g++ -O3 -std=c++20 -Wall -Wextra -Wpedantic -D__HIP_PLATFORM_AMD__ \
  study.cpp "$out/kernels.o" -lamdhip64 -o "$out/study"
echo "built $out/study"

# Optional vendor baseline: use the EXISTING user-validated AMD 7.1 overlay,
# not Canonical's known-broken RCCL. No package extraction or system changes.
if [ "${1:-}" = "--rccl" ]; then
  libdir="${ALICE_RCCL_LIB_DIR:-$HOME/scratch/rocm-debs/amd-7.1/opt/rocm-7.1.0/lib}"
  test -f "$libdir/librccl.so.1" || {
    echo "Missing validated RCCL overlay: $libdir; see research/rocm-overlay-method.md" >&2
    exit 1
  }
  g++ -O3 -std=c++20 -Wall -Wextra -Wpedantic -D__HIP_PLATFORM_AMD__ \
    "-DRCCL_EXPECTED_LIBRARY=\"$libdir/librccl.so.1\"" \
    rccl.cpp "$out/kernels.o" "$libdir/librccl.so.1" -lamdhip64 -ldl -o "$out/study-rccl"
  echo "built $out/study-rccl (run with LD_LIBRARY_PATH=$libdir)"
fi
