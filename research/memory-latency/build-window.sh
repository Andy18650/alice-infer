#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/memory-window/build"}
mkdir -p "$out";out=$(cd "$out" && pwd)
python3 "$here/generate-bursts.py" "$out/window-bursts.hpp"
cmp "$here/window-bursts.hpp" "$out/window-bursts.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra)
hipcc --version > "$out/compiler.txt"
printf '%q ' hipcc "${flags[@]}" "$here/window.hip" -o "$out/window" > "$out/build-command.txt";printf '\n' >> "$out/build-command.txt"
hipcc "${flags[@]}" "$here/window.hip" -o "$out/window"
/usr/bin/llvm-objdump-21 --offloading "$out/window" > "$out/offloading.txt"
for object in "$out"/window.*gfx90a; do
 /usr/bin/llvm-objdump-21 -d --demangle "$object" > "$object.isa.txt"
 /usr/bin/llvm-readobj-21 --notes "$object" > "$object.metadata.txt"
done
sha256sum "$out/window" "$here/window.hip" "$here/window-bursts.hpp" > "$out/hashes.txt"
