#!/usr/bin/env python3
"""Check branch cleanup preserves the ordered non-control instruction trace.

This is a small interpreter for generated labels, remaining-counter updates
and SCC branches, NOT an AMDGPU emulator. It compares every load/store/wait/
barrier/MFMA/pointer-update/clock instruction along the executed path. Real
GPU correctness tests and native ISA/resource checks remain separate gates.
"""

import argparse
import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE_COUNTS = (*range(13), 15, 16, 17, 31, 32, 33, 63, 64, 65,
                127, 128, 129, 255, 256, 257, 2048)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def asm_instructions(header):
    result = []
    for line in header.splitlines():
        match = re.match(r'^\s*"(.*)\\n\\t"$', line)
        if match:
            result.append(match[1])
    require(bool(result), 'No generated asm found')
    return result


def executed_trace(instructions, stage_count):
    labels = {op[:-1]: index for index, op in enumerate(instructions) if op.endswith(':')}
    remaining = stage_count
    scc = False
    pc = 0
    trace = []
    steps = 0
    while pc < len(instructions):
        steps += 1
        require(steps < 10000000, 'Control-flow interpreter did not terminate')
        op = instructions[pc]
        pc += 1
        if op.endswith(':'):
            continue
        opcode = op.split()[0]
        if opcode.startswith('s_cmp_'):
            require('%[remaining]' in op, f'Unknown comparison: {op}')
            rhs = int(op.rsplit(',', 1)[1])
            if opcode == 's_cmp_eq_u32':
                scc = remaining == rhs
            elif opcode == 's_cmp_le_u32':
                scc = remaining <= rhs
            elif opcode == 's_cmp_ge_u32':
                scc = remaining >= rhs
            elif opcode == 's_cmp_lt_u32':
                scc = remaining < rhs
            else:
                raise RuntimeError(f'Unsupported comparison: {op}')
        elif opcode == 's_sub_u32' and '%[remaining]' in op:
            remaining -= int(op.rsplit(',', 1)[1])
            require(remaining >= 0, 'Remaining counter underflow')
        elif opcode == 's_branch':
            pc = labels[op.split()[1]]
        elif opcode == 's_cbranch_scc1':
            if scc:
                pc = labels[op.split()[1]]
        else:
            trace.append(op)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-source', type=Path,
                        default=HERE / 'results/readable/source')
    parser.add_argument('--after-source', type=Path,
                        default=HERE / 'results/branch-clean/after/source')
    parser.add_argument('--work', type=Path,
                        default=ROOT / 'temp/native-gemm/branch-flow-audit')
    parser.add_argument('--output', type=Path,
                        default=HERE / 'results/branch-clean/checks')
    args = parser.parse_args()
    work = args.work.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh audit output directory')
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    proof = []

    for tile, stage_k, resident in itertools.product((128, 256), (32, 64), (False, True)):
        name = f't{tile}-s{stage_k}-r{int(resident)}'
        extra = ['--resident'] if resident else []
        headers = []
        for label, script in (('before', args.before_source / 'generate.py'),
                              ('after', args.after_source / 'generate.py')):
            file = work / f'{name}-{label}.hpp'
            subprocess.run(['python3', str(script), '--tile', str(tile), '--stage-k',
                            str(stage_k), *extra, '--output', str(file)], check=True)
            headers.append(file.read_text())
        if resident:
            require(headers[0] == headers[1], f'Resident control changed: {name}')
            proof.append(dict(configuration=name, resident_header_identical=True))
            continue

        before, after = (asm_instructions(header) for header in headers)
        checks = []
        for count in STAGE_COUNTS:
            old_trace = executed_trace(before, count)
            new_trace = executed_trace(after, count)
            require(old_trace == new_trace,
                    f'Non-control sequence changed: {name}, stages={count}')
            checks.append(dict(
                stages=count,
                executed_noncontrol_instructions=len(new_trace),
                trace_sha256=hashlib.sha256('\n'.join(new_trace).encode()).hexdigest(),
            ))
        proof.append(dict(configuration=name, checks=checks))
        print(name, f'{len(checks)} stage-count traces match exactly')

    (output / 'summary.json').write_text(json.dumps(proof, indent=2) + '\n')
    print('PASS: normal memory/math/wait/barrier traces unchanged; resident headers unchanged.')


if __name__ == '__main__':
    main()
