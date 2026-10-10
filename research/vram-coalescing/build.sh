#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
mkdir -p "$root/temp/vram-coalescing"
common="-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra"
for barriers in 0 1; do
  suffix=""
  if [ "$barriers" -eq 1 ]; then suffix="-barriers"; fi
  hipcc $common -DRESEARCH_SCHED_BARRIERS=$barriers "$here/probe.hip" \
    -o "$root/temp/vram-coalescing/default$suffix"
  # Same vectorization controls for every reader; inspect actual load widths.
  hipcc $common -DRESEARCH_SCHED_BARRIERS=$barriers \
    -fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false \
    "$here/probe.hip" -o "$root/temp/vram-coalescing/controlled$suffix"
done
hipcc $common -fno-vectorize -fno-slp-vectorize \
  -mllvm -amdgpu-load-store-vectorizer=false "$here/packing.hip" \
  -o "$root/temp/vram-coalescing/packing"
hipcc $common -DRESEARCH_SCHED_BARRIERS=0 "$here/ring.hip" \
  -o "$root/temp/vram-coalescing/ring"
echo "built barrier-free probes, matched barrier controls, packing, and ring probe"
