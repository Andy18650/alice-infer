#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/cu-load-ceiling"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/load_ceiling.hip" -o "$out/load-ceiling-probe"
llvm-objdump-21 --offloading "$out/load-ceiling-probe" > "$out/offloading.txt"
cp "$out/load-ceiling-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/load-ceiling-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/load-ceiling-probe.hsaco" > "$out/load-ceiling-probe.isa"
llvm-readobj-21 --notes "$out/load-ceiling-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/load_ceiling.hip" "$out/load-ceiling-probe" "$out/load-ceiling-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/load-ceiling-probe"
