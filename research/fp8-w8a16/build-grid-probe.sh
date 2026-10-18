#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/grid"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/grid_probe.hip" -o "$out/grid-probe"
llvm-objdump-21 --offloading "$out/grid-probe" > "$out/offloading.txt"
cp "$out/grid-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/grid-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/grid-probe.hsaco" > "$out/grid-probe.isa"
llvm-readobj-21 --notes "$out/grid-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/grid_probe.hip" "$out/grid-probe" "$out/grid-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/grid-probe"
