#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/fixed"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/fixed_probe.hip" -o "$out/fixed-probe"
llvm-objdump-21 --offloading "$out/fixed-probe" > "$out/offloading.txt"
cp "$out/fixed-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/fixed-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/fixed-probe.hsaco" > "$out/fixed-probe.isa"
llvm-readobj-21 --notes "$out/fixed-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/fixed_probe.hip" "$out/fixed-probe" "$out/fixed-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/fixed-probe"
