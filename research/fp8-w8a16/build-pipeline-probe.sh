#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/pipeline"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/pipeline_probe.hip" -o "$out/pipeline-probe"
llvm-objdump-21 --offloading "$out/pipeline-probe" > "$out/offloading.txt"
cp "$out/pipeline-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/pipeline-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/pipeline-probe.hsaco" > "$out/pipeline-probe.isa"
llvm-readobj-21 --notes "$out/pipeline-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/pipeline_probe.hip" "$out/pipeline-probe" "$out/pipeline-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/pipeline-probe"
