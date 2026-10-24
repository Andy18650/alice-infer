#!/usr/bin/env python3
"""Check the FP8 producer change preserves compute/raw-read/fence ordering.

Loads, stores, source-pointer steps and conversion deliberately differ. This
projection does NOT assert their equivalence or emulate their values/latency;
independent packing, GPU expansion and whole-GEMM references are separate gates.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, executed_trace, require


KEEP = ('v_mfma_', 'ds_read_b128', 's_waitcnt', 's_barrier', 's_memtime', 's_memrealtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/fp8-b-audit')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    for name, source, extra in (
        ('before', HERE / 'results/mfma-first/after/source/generate.py', ['--tile', '128', '--stage-k', '32']),
        ('after', HERE / 'results/fp8-b/after/source/generate_fp8.py', []),
    ):
        file = output / f'{name}.hpp'
        subprocess.run(['python3', str(source), *extra, '--output', str(file)], check=True)
        # Do not interpret the separate conversion-check helper asm as part
        # of the pipeline (it is a different, uncalled device function here).
        header = file.read_text().split('__device__ __forceinline__ uint2 expand_fp8_for_check')[0]
        traces.append(asm_instructions(header))
    proof = []
    for count in STAGE_COUNTS:
        before, after = ([op for op in executed_trace(code, count) if op.startswith(KEEP)]
                         for code in traces)
        require(before == after, f'Compute/raw-read/fence order changed: stages={count}')
        proof.append(dict(stages=count, projected_instructions=len(after),
                          sha256=hashlib.sha256('\n'.join(after).encode()).hexdigest()))

    loops = [(ROOT / f'temp/native-gemm/{folder}/s32-t128/loop.txt').read_text().splitlines()
             for folder in ('mfma-first', 'fp8-b')]
    native_math = [[op for op in loop if op.startswith('v_mfma_')] for loop in loops]
    require(native_math[0] == native_math[1], 'Native MFMA operand/order changed')
    reads = []
    for loop in loops:
        # C++ may assign different physical address-input VGPRs. Normalize
        # only those addresses; raw destination registers and offsets stay exact.
        addresses = {}
        projection = []
        for op in loop:
            if op.startswith('ds_read_b128'):
                match = re.fullmatch(r'(ds_read_b128 v\[\d+:\d+\], )(v\d+)(.*)', op)
                require(match is not None, 'Unexpected LDS-read syntax')
                address = addresses.setdefault(match[2], f'addr{len(addresses)}')
                projection.append(match[1] + address + match[3])
        require(len(addresses) == 2, 'Expected separate A/B LDS input addresses')
        reads.append(projection)
    require(reads[0] == reads[1], 'Native raw destinations/offset/order changed')
    (output / 'proof.json').write_text(json.dumps(dict(
        generated=proof, native_math_identical=True, native_reads_address_normalized_equal=True,
        before_loop_instructions=len(loops[0]), after_loop_instructions=len(loops[1]),
    ), indent=2) + '\n')
    print(f'PASS {len(proof)} stage counts: compute/raw-read/fence projection exact; native math/read order matches.')


if __name__ == '__main__':
    main()
