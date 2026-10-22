#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd);root=$(cd "$here/../.." && pwd)
if (($#>1));then echo 'usage: build-grid-vgpr.sh [output-directory]' >&2;exit 2;fi
out=${1:-"$root/temp/lds-staging/grid-vgpr"};mkdir -p "$out/tmp";out=$(cd "$out" && pwd);export TMPDIR="$out/tmp"
python3 "$here/generate-vgpr.py" --output "$out/vgpr-loads.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra -I"$out")
{ printf '%q' hipcc;printf ' %q' "${flags[@]}";printf '\n'; } > "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/grid-vgpr.hip" -o "$out/probe"
llvm-objdump-21 --offloading "$out/probe" > "$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" > "$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" > "$out/metadata.txt"
python3 - "$here" "$out" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json,sys
from pathlib import Path
here,out,cc=sys.argv[1:4];src=str(Path(here)/'grid-vgpr.hip')
Path(out,'compile_commands.json').write_text(json.dumps([dict(directory=out,file=src,arguments=[cc,*sys.argv[4:],'-x','hip','-c',src])],indent=2)+'\n')
PY
python3 "$here/inspect_vgpr.py" "$out"
sha256sum "$here/grid-vgpr.hip" "$here/generate-vgpr.py" "$out/vgpr-loads.hpp" "$out/probe" "$out/probe.hsaco" > "$out/hashes.txt"
echo "Built $out/probe"
