#!/bin/sh
# Inspect an actual ceiling executable and test a timestamped private twin.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
binary="${1:-$root/temp/vram-ceiling/final-1/ceiling}"
out="${2:-$root/temp/vram-ceiling/occupancy/reproduce}"
[ -f "$binary" ] || { echo "Missing ceiling executable: $binary" >&2; exit 1; }
mkdir -p "$out"
cp "$binary" "$out/original"
cp "$here/occupancy.cpp" "$here/occupancy-timeline.hip" "$out/"
{
  echo "O3 HIP C++20 gfx90a; HIP device 0; CPU+memory NUMA node 0; no model/context"
  echo "native eager launches; no holds/barriers/atomics/cooperative launch or clock overrides"
  hipcc --version
  sha256sum "$out/original" "$here/occupancy.cpp" "$here/occupancy-timeline.hip"
} > "$out/metadata.txt"
llvm-objdump --offloading "$out/original" > "$out/offloading.txt"
object="$out/original.0.hipv4-amdgcn-amd-amdhsa--gfx90a"
llvm-readelf --notes "$object" > "$out/original-notes.txt"
hipcc -O3 -std=c++20 -Wall -Wextra "$here/occupancy.cpp" -o "$out/query"
numactl --cpunodebind=0 --membind=0 "$out/query" "$object" > "$out/exact-query.txt"
hipcc -O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra "$here/occupancy-timeline.hip" -o "$out/timeline-probe"
sha256sum "$out/query" "$out/timeline-probe" >> "$out/metadata.txt"
rocm-smi --showbus --showuse --showclocks > "$out/smi-before.txt"
for sample in a b; do
  numactl --cpunodebind=0 --membind=0 "$out/timeline-probe" "$out/timeline-$sample" \
    > "$out/timeline-$sample.log" 2>&1
done
rocm-smi --showbus --showuse --showclocks > "$out/smi-after.txt"
echo "saved $out"
