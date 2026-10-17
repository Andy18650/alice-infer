#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
python3 "$here/check_calculator.py"
bash "$here/build.sh" "${OUT:-$root/temp/fp8-w8a16}"
python3 "$here/audit.py" "${OUT:-$root/temp/fp8-w8a16}"
numactl --cpunodebind=0 --membind=0 "${OUT:-$root/temp/fp8-w8a16}/probe" "${STEPS:-4096}" "${ROUNDS:-9}"
