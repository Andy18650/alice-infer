#!/usr/bin/env python3
"""Verify exactly one existing MFMA moves after each non-final stage's VM0.

The first VM0 is startup and must not get an MFMA: operands have not yet been
seeded. Every subsequent VM0 in an executed normal trace is a stage head.
The expected transformation moves its first following MFMA to immediately
post-VM0 and leaves every other instruction in its prior order. This is a
control/data-order check, not a GPU latency model or hardware emulator.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, executed_trace, require


def expected_move(trace, skip_startup=True):
    result = list(trace)
    heads = [index for index, op in enumerate(trace) if op == 's_waitcnt vmcnt(0)']
    if skip_startup:
        heads = heads[1:]
    for head in heads:
        matrix_index = next(index for index in range(head + 1, len(result))
                            if result[index].startswith('v_mfma_'))
        instruction = result.pop(matrix_index)
        result.insert(head + 1, instruction)
    return result, len(heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-source', type=Path,
                        default=HERE / 'results/branch-clean/after/source')
    parser.add_argument('--after-source', type=Path, default=HERE)
    parser.add_argument('--work', type=Path, default=ROOT / 'temp/native-gemm/mfma-first/audit')
    parser.add_argument('--output', type=Path, default=HERE / 'results/mfma-first-audit')
    args = parser.parse_args()
    output = args.output.resolve()
    work = args.work.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh audit output directory')
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    proofs = []

    for tile, stage, resident in itertools.product((128, 256), (32, 64), (False, True)):
        name = f't{tile}-s{stage}-r{int(resident)}'
        headers = []
        extra = ['--resident'] if resident else []
        for label, source in (('before', args.before_source), ('after', args.after_source)):
            file = work / f'{name}-{label}.hpp'
            subprocess.run(['python3', str(source / 'generate.py'), '--tile', str(tile),
                            '--stage-k', str(stage), *extra, '--output', str(file)], check=True)
            headers.append(file.read_text())
        if resident:
            require(headers[0] == headers[1], f'Resident diagnostic changed: {name}')
            proofs.append(dict(configuration=name, resident_header_identical=True))
            continue

        before, after = (asm_instructions(header) for header in headers)
        cases = []
        for count in STAGE_COUNTS:
            old_trace = executed_trace(before, count)
            new_trace = executed_trace(after, count)
            expected, moves = expected_move(old_trace)
            require(moves == max(0, count - 1), 'Unexpected number of stage heads')
            require(expected == new_trace,
                    f'Change exceeds the requested move: {name}, stages={count}')
            cases.append(dict(
                stages=count,
                moved_mfmas=moves,
                trace_sha256=hashlib.sha256('\n'.join(new_trace).encode()).hexdigest(),
            ))
        proofs.append(dict(configuration=name, checks=cases))
        print(name, f'{len(cases)} exact single-move traces match')

    native = []
    for tile in (128, 256):
        old = (ROOT / f'temp/native-gemm/branch-clean/s32-t{tile}/loop.txt').read_text().splitlines()
        new = (ROOT / f'temp/native-gemm/mfma-first/s32-t{tile}/loop.txt').read_text().splitlines()
        expected, moves = expected_move(old, skip_startup=False)
        require(expected == new and moves == 2, f'Unexpected native loop change: tile{tile}')
        native.append(dict(
            tile=tile,
            moved_mfmas=moves,
            instruction_count=len(new),
            sha256=hashlib.sha256('\n'.join(new).encode()).hexdigest(),
        ))
    (output / 'equivalence.json').write_text(
        json.dumps(dict(generated=proofs, native=native), indent=2) + '\n'
    )
    print('PASS: exactly one MFMA move/stage; all other order and resident headers unchanged.')


if __name__ == '__main__':
    main()
