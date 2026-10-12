#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
mkdir -p "$root/temp/gemv-layout"
for strong in 0 1; do
  suffix=""; [ "$strong" -eq 0 ] || suffix="-strong"
  hipcc -O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra -pthread \
    -DGEMV_LAST_STRONG_PUBLICATION=$strong "$here/bench.hip" -o "$root/temp/gemv-layout/bench$suffix"
done
echo "built release-publication and strong-control benchmarks"
