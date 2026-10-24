#!/usr/bin/env python3
"""Check the two FP8 scheduling increments without pretending to model latency."""

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, executed_trace, require


KEEP = ('v_mfma_', 'ds_read_b128', 's_waitcnt', 's_barrier', 's_memtime', 's_memrealtime')
ARITHMETIC = ('v_pk_ashrrev_i16', 'v_and_b32', 'v_pk_mul_f16')


def move_current_c01(trace, skip_startup):
    result = list(trace)
    heads = [index for index, op in enumerate(trace) if op == 's_waitcnt vmcnt(0)']
    if skip_startup:
        heads = heads[1:]
    for head in heads:
        first = next(i for i in range(head + 1, len(result)) if result[i].startswith('v_mfma_'))
        second = next(i for i in range(first + 1, len(result)) if result[i].startswith('v_mfma_'))
        arithmetic = [i for i in range(first + 1, second) if result[i].startswith(ARITHMETIC)]
        require(len(arithmetic) == 24, 'Expected full24-op arithmetic block before C01')
        moved = result.pop(second)
        result.insert(arithmetic[5] + 1, moved)
    return result, len(heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/fp8-schedule-audit')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    code = {}
    for name in ('burst', 'early', 'interleaved'):
        file = output / f'{name}.hpp'
        if name == 'burst':
            command = ['python3', str(HERE / 'results/fp8-b/after/source/generate_fp8.py')]
        else:
            command = ['python3', str(HERE / 'generate_fp8.py'), '--schedule', name]
        subprocess.run([*command, '--output', str(file)], check=True)
        pipeline = file.read_text().split('__device__ __forceinline__ uint2 expand_fp8_for_check')[0]
        code[name] = asm_instructions(pipeline)

    regenerated = output / 'regenerated-burst.hpp'
    subprocess.run(['python3', str(HERE / 'generate_fp8.py'), '--schedule', 'burst',
                    '--output', str(regenerated)], check=True)
    pipeline = regenerated.read_text().split('__device__ __forceinline__ uint2 expand_fp8_for_check')[0]
    require(asm_instructions(pipeline) == code['burst'], 'Historical burst instruction flow changed')
    checks = []
    for count in STAGE_COUNTS:
        traces = {name: executed_trace(instructions, count) for name, instructions in code.items()}
        projections = [[op for op in trace if op.startswith(KEEP)] for trace in traces.values()]
        require(projections[0] == projections[1] == projections[2],
                f'Compute/raw-read/fence projection changed: stages={count}')
        for name, trace in traces.items():
            counts = collections.Counter(op.split()[0] for op in trace)
            for opcode, expected in (('global_load_dwordx4', 3 * count),
                                     ('ds_write_b128', 4 * count), ('v_perm_b32', 8 * count)):
                require(counts[opcode] == expected, f'{name}: over/under-fetch or conversion count')
        expected, moves = move_current_c01(traces['early'], skip_startup=True)
        require(expected == traces['interleaved'] and moves == max(0, count - 1),
                f'Interleave change exceeds one current C01 move: stages={count}')
        trace = traces['early']
        b_loads = [i for i, op in enumerate(trace)
                   if op.startswith('global_load_dwordx4 v[60:63]')]
        # Initial and (if present) second stage load in startup; all subsequent
        # B prefetches immediately follow all eight copies, before arithmetic.
        for index in b_loads[2:]:
            require(all(op.startswith('v_perm_b32') for op in trace[index - 8:index]),
                    'Future B load not immediately after all eight copies')
        checks.append(dict(stages=count, moved_current_mfmas=moves,
                           early_future_b_loads=max(0, len(b_loads) - 2),
                           sha256=hashlib.sha256('\n'.join(traces['interleaved']).encode()).hexdigest()))

    loops = [(ROOT / f'temp/native-gemm/fp8-schedule/{name}/loop.txt').read_text().splitlines()
             for name in ('early', 'interleaved')]
    expected, moves = move_current_c01(loops[0], skip_startup=False)
    require(expected == loops[1] and moves == 2, 'Native loop differs beyond two C01 moves')
    inventories = [json.loads((ROOT / f'temp/native-gemm/fp8-schedule/{name}/inventory.json').read_text())
                   for name in ('early', 'interleaved')]
    for key in ('vgpr_count', 'agpr_count', 'sgpr_count', 'private_segment_fixed_size', 'lds_bytes'):
        require(inventories[0][key] == inventories[1][key], f'Interleave resource change: {key}')
    (output / 'proof.json').write_text(json.dumps(dict(
        checks=checks, native_exact_two_mfma_moves=True, native_instructions=len(loops[1]),
        same_interleave_resources=True, regenerated_burst_asm_identical=True,
    ), indent=2) + '\n')
    print(f'PASS {len(checks)} stage counts: no overfetch, fixed math/read/fences, exact C01 move; native2 moves/resources match.')


if __name__ == '__main__':
    main()
