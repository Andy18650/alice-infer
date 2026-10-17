#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/load"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/load_probe.hip" -o "$out/load-probe"
llvm-objdump-21 --offloading "$out/load-probe" > "$out/offloading.txt"
cp "$out/load-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/load-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/load-probe.hsaco" > "$out/load-probe.isa"
llvm-readobj-21 --notes "$out/load-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/load_probe.hip" "$out/load-probe" "$out/load-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/load-probe"
