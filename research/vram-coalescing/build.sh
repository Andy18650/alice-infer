#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
mkdir -p "$root/temp/vram-coalescing"
common="-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra"
hipcc $common "$here/probe.hip" -o "$root/temp/vram-coalescing/default"
# Apply the same controls to all three readers. A's explicit vector load
# remains a vector; B/C scalar source loads must be checked in disassembly.
hipcc $common -fno-vectorize -fno-slp-vectorize \
  -mllvm -amdgpu-load-store-vectorizer=false "$here/probe.hip" \
  -o "$root/temp/vram-coalescing/controlled"
hipcc $common -fno-vectorize -fno-slp-vectorize \
  -mllvm -amdgpu-load-store-vectorizer=false "$here/packing.hip" \
  -o "$root/temp/vram-coalescing/packing"
echo "built default, controlled, and packing probes"
