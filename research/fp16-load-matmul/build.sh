#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
depth=${1:-8}; unroll=${2:-32}; lds_stride=${4:-1}; ready_banks=${5:-2}
out=${3:-"$root/temp/fp16-load-matmul/d$depth-u$unroll-lds$lds_stride-verified-r$ready_banks-waited"}
mkdir -p "$out/tmp"; out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
python3 "$here/generate.py" --depth "$depth" --unroll "$unroll" --lds-stride "$lds_stride" --ready-banks "$ready_banks" --output "$out/pipeline.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra -I"$out")
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; printf '\n'; } > "$out/compiler-flags.txt"
python3 - "$out" "$here" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json,sys
from pathlib import Path
out,here,compiler=sys.argv[1:4]
source=str(Path(here)/'probe.hip')
entry={'directory':out,'file':source,'arguments':[compiler,*sys.argv[4:],'-x','hip','-c',source]}
Path(out,'compile_commands.json').write_text(json.dumps([entry],indent=2)+'\n')
PY
hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe"
llvm-objdump-21 --offloading "$out/probe" > "$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" > "$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/probe.hip" "$here/verification.hpp" "$here/generate.py" "$out/pipeline.hpp" "$out/probe" "$out/probe.hsaco" > "$out/hashes.txt"
echo "Built $out/probe"
