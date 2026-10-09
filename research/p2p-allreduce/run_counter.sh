#!/bin/sh
# Bounded counter-only sweep; never continues after a failure/timeout.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:-$root/temp/p2p-allreduce/counter-results}"
target="${2:-10000}"
samples="${3:-9}"
"$here/build.sh" --counter
mkdir -p "$out"
{
  echo "O3 C++20 gfx90a; CPU+memory NUMA node 0; no payload/attention context"
  echo "target=$target timed increments/rank; warmup=64; samples=$samples"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$here/counter.hip" "$root/temp/p2p-allreduce/counter"
} > "$out/metadata.txt"
first=1
for ws in 1 2 4 8; do
  for poll in serial parallel neighbor; do
    for order in relaxed release; do
      tag="ws${ws}-${poll}-${order}"
      timeout 30s numactl --cpunodebind=0 --membind=0 \
        "$root/temp/p2p-allreduce/counter" "$ws" "$target" "$samples" 0 "$poll" "$order" 64 \
        > "$out/$tag.csv" 2> "$out/$tag.log"
      if [ "$first" = 1 ]; then head -1 "$out/$tag.csv" > "$out/results.csv"; first=0; fi
      tail -1 "$out/$tag.csv" >> "$out/results.csv"
    done
  done
done
# GCDs 1/2 cross cards; 0/1 above is the intra-card pair.
for poll in serial parallel; do
  for order in relaxed release; do
    tag="ws2-base1-${poll}-${order}"
    timeout 30s numactl --cpunodebind=0 --membind=0 \
      "$root/temp/p2p-allreduce/counter" 2 "$target" "$samples" 1 "$poll" "$order" 64 \
      > "$out/$tag.csv" 2> "$out/$tag.log"
    tail -1 "$out/$tag.csv" >> "$out/results.csv"
  done
done
echo "saved $out/results.csv"
