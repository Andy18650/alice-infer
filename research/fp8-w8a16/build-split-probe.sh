#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/split"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/split_probe.hip" -o "$out/split-probe"
llvm-objdump-21 --offloading "$out/split-probe" > "$out/offloading.txt"
cp "$out/split-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/split-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/split-probe.hsaco" > "$out/split-probe.isa"
llvm-readobj-21 --notes "$out/split-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/split_probe.hip" "$out/split-probe" "$out/split-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/split-probe"
