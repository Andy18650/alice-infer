#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
mkdir -p "$root/temp/vram-coalescing"
common="-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra"
for policy in off phase all; do
  case "$policy" in
    off) suffix=""; barriers=0; load_barriers=0;;
    phase) suffix="-phase-barriers"; barriers=1; load_barriers=0;;
    all) suffix="-barriers"; barriers=1; load_barriers=1;;
  esac
  hipcc $common -DRESEARCH_SCHED_BARRIERS=$barriers -DRESEARCH_LOAD_BARRIERS=$load_barriers "$here/probe.hip" \
    -o "$root/temp/vram-coalescing/default$suffix"
  # Same vectorization controls for every reader; inspect actual load widths.
  hipcc $common -DRESEARCH_SCHED_BARRIERS=$barriers -DRESEARCH_LOAD_BARRIERS=$load_barriers \
    -fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false \
    "$here/probe.hip" -o "$root/temp/vram-coalescing/controlled$suffix"
done
hipcc $common -fno-vectorize -fno-slp-vectorize \
  -mllvm -amdgpu-load-store-vectorizer=false "$here/packing.hip" \
  -o "$root/temp/vram-coalescing/packing"
hipcc $common -DRESEARCH_SCHED_BARRIERS=0 "$here/ring.hip" \
  -o "$root/temp/vram-coalescing/ring"
${CXX:-c++} -O2 -std=c++20 -Wall -Wextra "$here/clock-monitor.cpp" -lrocm_smi64 \
  -o "$root/temp/vram-coalescing/clock-monitor"
echo "built barrier-free, phase-only and per-load barrier probes, packing, ring, and clock monitor"
