#!/bin/sh
# Optional vendor baseline; one communicator initialization per world size.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
libdir="${ALICE_RCCL_LIB_DIR:-$HOME/scratch/rocm-debs/amd-7.1/opt/rocm-7.1.0/lib}"
out="${1:-$root/temp/p2p-allreduce/rccl}"
slots="${2:-64}"
samples="${3:-15}"
"$here/build.sh" --rccl
mkdir -p "$out"
export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
{
  echo "O3 C++20; CPU and memory NUMA node 0; context=n/a (collective only)"
  echo "out-of-place; slots=$slots samples=$samples; graph and eager; auto RCCL algorithm"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$root/temp/p2p-allreduce/study-rccl" "$libdir/librccl.so.1" \
    "$here/rccl.cpp" "$here/kernels.hip" "$here/kernels.hpp"
} > "$out/metadata.txt"
for ws in 4 8; do
  # No blind continuation after a timeout/failure; set -e stops the study.
  timeout 120s numactl --cpunodebind=0 --membind=0 \
    "$root/temp/p2p-allreduce/study-rccl" "$ws" "$slots" "$samples" 0 both \
    > "$out/ws${ws}.csv" 2> "$out/ws${ws}.log"
  echo "saved $out/ws${ws}.csv"
done
