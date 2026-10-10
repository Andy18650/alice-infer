#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
samples="${2:-7}"
binary="$root/temp/vram-coalescing/packing"
if [ ! -x "$binary" ] || [ "$binary" -ot "$here/packing.hip" ] || [ "$binary" -ot "$here/readers.hpp" ]; then
  echo "Missing/stale packing probe: run $here/build.sh first" >&2; exit 1
fi
mkdir -p "$out"
cp "$here/packing.hip" "$here/readers.hpp" "$here/packing-summary.py" "$out/"
cp "$binary" "$out/packing"
{
  echo "O3 HIP C++20 gfx90a; HIP device 0; CPU+memory NUMA node 0; no model/context"
  echo "1 GiB, chunks=4, block=256; grids=52/104/208/416/832/1664; samples=$samples"
  echo "-fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false"
  echo "CU/SIMD/slot decoder: gfx90a HW_ID register 4; no holds/atomics/cooperative launch or clock overrides"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$out/packing.hip" "$out/readers.hpp" "$out/packing" "$out/packing-summary.py"
} > "$out/metadata.txt"
llvm-objdump --offloading "$out/packing" > "$out/offloading.txt"
llvm-objdump -d --demangle "$out/packing.0.hipv4-amdgcn-amd-amdhsa--gfx90a" > "$out/disassembly.txt"
rocm-smi --showbus --showuse --showclocks > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/packing" "$out" "$samples" > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/../vram-ceiling/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
rocm-smi --showbus --showuse --showclocks > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
python3 "$here/packing-summary.py" "$out"
echo "saved $out"
