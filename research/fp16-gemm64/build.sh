#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
depth=${1:-2}; unroll=${2:-4}; stage=${3:-64}
out=${4:-"$root/temp/fp16-gemm64/d$depth-u$unroll-t$stage-direct-only"}
mkdir -p "$out/tmp"; out=$(cd "$out" && pwd); export TMPDIR="$out/tmp"
python3 "$here/generate.py" --depth "$depth" --unroll "$unroll" --stage-k "$stage" --output "$out/compute.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra -I"$out")
{ printf '%q' hipcc;printf ' %q' "${flags[@]}";printf '\n'; } > "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/gemm64.hip" -o "$out/probe"
llvm-objdump-21 --offloading "$out/probe" > "$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" > "$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" > "$out/metadata.txt"
python3 - "$out" "$here" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json,sys
from pathlib import Path
out,here,cc=sys.argv[1:4];src=str(Path(here)/'gemm64.hip')
Path(out,'compile_commands.json').write_text(json.dumps([dict(directory=out,file=src,arguments=[cc,*sys.argv[4:],'-x','hip','-c',src])],indent=2)+'\n')
PY
sha256sum "$here/gemm64.hip" "$here/data.hpp" "$here/generate.py" "$out/compute.hpp" "$out/probe" "$out/probe.hsaco" > "$out/hashes.txt"
echo "Built $out/probe"
