#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/memory-latency/build"}
mkdir -p "$out";out=$(cd "$out" && pwd)
hipcc --version > "$out/compiler.txt"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra)
printf '%q ' hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe" > "$out/build-command.txt"
printf '\n' >> "$out/build-command.txt"
hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe"
/usr/bin/llvm-objdump-21 --offloading "$out/probe" > "$out/offloading.txt"
for object in "$out"/probe.*gfx90a; do
  /usr/bin/llvm-objdump-21 -d --demangle "$object" > "$object.isa.txt"
  /usr/bin/llvm-readobj-21 --notes "$object" > "$object.metadata.txt"
done
sha256sum "$out/probe" "$here/probe.hip" > "$out/hashes.txt"
