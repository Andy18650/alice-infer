#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
mkdir -p "$root/temp/vram-addressing"
hipcc -O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra "$here/probe.hip" \
  -o "$root/temp/vram-addressing/probe"
echo "built $root/temp/vram-addressing/probe"
