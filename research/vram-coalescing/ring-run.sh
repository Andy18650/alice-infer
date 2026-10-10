#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
original="${2:-$root/temp/vram-ceiling/final-1/ceiling}"
rounds="${3:-9}"
binary="$root/temp/vram-coalescing/ring"
if [ ! -x "$binary" ] || [ "$binary" -ot "$here/ring.hip" ] || [ "$binary" -ot "$here/readers.hpp" ]; then
  echo "Missing/stale ring probe: run $here/build.sh first" >&2; exit 1
fi
[ -f "$original" ] || { echo "Missing historical executable: $original" >&2; exit 1; }
mkdir -p "$out"
cp "$here/ring.hip" "$here/readers.hpp" "$out/"
cp "$binary" "$out/ring"
cp "$original" "$out/original"
llvm-objdump --offloading "$out/original" > "$out/offloading-original.txt"
object="$out/original.0.hipv4-amdgcn-amd-amdhsa--gfx90a"
llvm-objdump --offloading "$out/ring" > "$out/offloading-ring.txt"
llvm-objdump -d --demangle "$out/ring.0.hipv4-amdgcn-amd-amdhsa--gfx90a" > "$out/disassembly.txt"
{
  echo "O3 HIP C++20 gfx90a; device 0; CPU+memory NUMA node 0; no model/context"
  echo "free/ordered ring depths at 104x256; batch4 controls; exact historical 832x256 uint2/U16"
  echo "1/2/4 GiB jointly shuffled; rounds=$rounds; 12 reads/batch; ordering constraints only on labeled ordered variants; no clock overrides"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$out/ring.hip" "$out/readers.hpp" "$out/ring" "$out/original" "$object"
} > "$out/metadata.txt"
rocm-smi --showbus --showuse --showclocks > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/ring" "$out" "$object" "$rounds" > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/../vram-ceiling/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
rocm-smi --showbus --showuse --showclocks > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
python3 "$here/summarize.py" "$out" > "$out/summary.txt"
echo "saved $out"
