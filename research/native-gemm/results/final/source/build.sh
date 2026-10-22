#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd);root=$(cd "$here/../.." && pwd)
if (($#>4));then echo 'usage: build.sh [128|256 tile] [output-directory] [gemm|resident] [32|64 StageK]' >&2;exit 2;fi
tile=${1:-128};case "$tile" in 128|256);;*) echo 'tile must be128 or256' >&2;exit 2;;esac
kind=${3:-gemm};extra=();case "$kind" in gemm);;resident)extra=(--resident);;*)echo 'kind must be gemm or resident' >&2;exit 2;;esac
stage=${4:-32};case "$stage" in 32|64);;*)echo 'StageK must be32 or64' >&2;exit 2;;esac
out=${2:-"$root/temp/native-gemm/s$stage-t$tile"};mkdir -p "$out/tmp";out=$(cd "$out" && pwd);export TMPDIR="$out/tmp"
python3 "$here/generate.py" --tile "$tile" --stage-k "$stage" "${extra[@]}" --output "$out/compute.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra -I"$out")
{ printf '%q' hipcc;printf ' %q' "${flags[@]}";printf '\n'; } > "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe"
llvm-objdump-21 --offloading "$out/probe" > "$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" > "$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" > "$out/metadata.txt"
python3 - "$here" "$out" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json,sys
from pathlib import Path
here,out,cc=sys.argv[1:4];src=str(Path(here)/'probe.hip')
Path(out,'compile_commands.json').write_text(json.dumps([dict(directory=out,file=src,arguments=[cc,*sys.argv[4:],'-x','hip','-c',src])],indent=2)+'\n')
PY
python3 "$here/inspect_code.py" "$out"
sha256sum "$here/probe.hip" "$here/data.hpp" "$here/generate.py" "$out/compute.hpp" "$out/probe" "$out/probe.hsaco" > "$out/hashes.txt"
echo "Built $out/probe"
