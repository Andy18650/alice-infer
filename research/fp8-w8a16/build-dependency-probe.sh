#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/dependency"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/dependency_probe.hip" -o "$out/dependency-probe"
llvm-objdump-21 --offloading "$out/dependency-probe" > "$out/offloading.txt"
cp "$out/dependency-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/dependency-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/dependency-probe.hsaco" > "$out/dependency-probe.isa"
llvm-readobj-21 --notes "$out/dependency-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/dependency_probe.hip" "$out/dependency-probe" "$out/dependency-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/dependency-probe"
