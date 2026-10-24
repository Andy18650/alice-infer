#!/usr/bin/env python3
"""Expand only the emitted packed operations and compare exact scalar semantics.

Native branch targets are compared in expanded instruction coordinates, not
raw byte displacements. This is an instruction/order check, not a GPU emulator.
Actual scale/bank ownership checks and GPU references are separate gates.
"""

import argparse
import json
import re
from pathlib import Path

from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, require
from audit_scaled import execute
from inspect_code import parse_instructions


PAIR = r'v\[(\d+):(\d+)\]'


def expand(op):
    if op.startswith('v_pk_mov_b32'):
        operand = r'(0|s\[\d+:\d+\])'
        match = re.fullmatch(r'v_pk_mov_b32 ' + PAIR + ', ' + operand + ', ' + operand + r'(.*)', op)
        require(match is not None, 'Unexpected packed move syntax')
        low, high = int(match[1]), int(match[2])
        require(low % 2 == 0 and high == low + 1, 'Unaligned move destination')
        if (match[3], match[4]) == ('0', '0'):
            return [f'v_mov_b32 v{low}, 0', f'v_mov_b32 v{high}, 0']
        require(match[3] == match[4], 'Expected identical SGPR pair move sources')
        source = re.fullmatch(r's\[(\d+):(\d+)\]', match[3])
        require(source is not None, 'Unexpected packed move source')
        a, b = map(int, source.groups())
        require(a % 2 == 0 and b == a + 1, 'Unaligned move source')
        # V_PK_MOV uses its two op_sel bits for the two destination words;
        # unlike packed arithmetic, default is low source word for BOTH.
        selectors = re.search(r'\bop_sel:\[(\d),(\d)\]', match[5])
        selectors = tuple(map(int, selectors.groups())) if selectors else (0, 0)
        return [f'v_mov_b32 v{low}, s{a + selectors[0]}',
                f'v_mov_b32 v{high}, s{a + selectors[1]}']
    if op.startswith('v_pk_fma_f32'):
        match = re.fullmatch(r'v_pk_fma_f32 ' + PAIR + ', ' + PAIR
                             + r', s\[14:15\], ' + PAIR + r'(.*)', op)
        require(match is not None, 'Unexpected packed FMA syntax')
        destination, destination_hi, source, source_hi, accumulator, accumulator_hi = map(int, match.groups()[:6])
        require(destination == accumulator and destination_hi == accumulator_hi,
                'FMA not updating the same scaled accumulator')
        require(destination_hi == destination + 1 and source_hi == source + 1
                and destination % 2 == 0 and source % 2 == 0, 'Unaligned FMA pair')
        modifiers = match[7]
        low = re.search(r'\bop_sel:\[(\d),(\d),(\d)\]', modifiers)
        high = re.search(r'\bop_sel_hi:\[(\d),(\d),(\d)\]', modifiers)
        low = tuple(map(int, low.groups())) if low else (0, 0, 0)
        high = tuple(map(int, high.groups())) if high else (1, 1, 1)
        require(low[0] == low[2] == 0 and high[0] == high[2] == 1,
                'Packed FMA is not processing consecutive vector elements')
        require(low[1] == high[1], 'Different scales selected for packed halves')
        scale = 14 + low[1]
        return [f'v_fma_f32 v{destination + i}, v{source + i}, s{scale}, v{destination + i}'
                for i in range(2)]
    return [op.replace('v_mov_b32_e32', 'v_mov_b32')]


def native_normalized(path):
    instructions = parse_instructions(path.read_text())
    end = next(i for i, (_, op) in enumerate(instructions) if op == 's_endpgm')
    instructions = instructions[:end + 1]
    offsets = {}
    output = []
    branches = []
    for pc, op in instructions:
        offsets[pc] = len(output)
        if op.startswith(('s_branch ', 's_cbranch_')):
            opcode, immediate = op.split()
            displacement = int(immediate, 0)
            if displacement >= 32768:
                displacement -= 65536
            branches.append((len(output), opcode, pc + 4 + displacement * 4))
            output.append(None)
        else:
            output.extend(expand(op))
    for index, opcode, target in branches:
        require(target in offsets, 'Native branch target missing')
        output[index] = f'{opcode} instruction:{offsets[target]}'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/scaled-packed-fixed-audit')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), 'Choose a fresh output')
    output.mkdir(parents=True, exist_ok=True)
    builds = {
        'scalar': ROOT / 'temp/native-gemm/scaled/scaled',
        'mov': ROOT / 'temp/native-gemm/scaled-packed-fixed/scaled-mov',
        'packed': ROOT / 'temp/native-gemm/scaled-packed-fixed/scaled-all',
    }
    require(expand('v_pk_mov_b32 v[2:3], s[4:5], s[4:5]')
            != ['v_mov_b32 v2, s4', 'v_mov_b32 v3, s5'],
            'Auditor must reject the original incorrect default selector assumption')
    source_projections = {}
    native_projections = {}
    ownership = {}
    for variant, build in builds.items():
        code = asm_instructions((build / 'compute.hpp').read_text())
        unpacked = [item for op in code for item in expand(op)]
        source_projections[variant] = unpacked
        native_projections[variant] = native_normalized(build / 'probe.isa')
        ownership[variant] = [execute(unpacked, count) for count in STAGE_COUNTS]
    inventory = {variant: json.loads((build / 'inventory.json').read_text())
                 for variant, build in builds.items()}
    for variant in ('mov', 'packed'):
        for field in ('vgpr_count', 'agpr_count', 'sgpr_count', 'private_segment_fixed_size', 'lds_bytes'):
            require(inventory[variant][field] == inventory['scalar'][field],
                    f'Resource change: {variant}/{field}')
        require(source_projections[variant] == source_projections['scalar'],
                f'Generated changes exceed packed replacements: {variant}')
        require(native_projections[variant] == native_projections['scalar'],
                f'Native changes exceed packed replacements: {variant}')
        require(ownership[variant] == ownership['scalar'], 'Scale/bank semantics changed')
    (output / 'proof.json').write_text(json.dumps(dict(
        generated_full_flow_equal_after_expansion=True,
        native_full_kernel_equal_after_expansion_and_branch_normalization=True,
        ownership=ownership, identical_resources=True,
        expanded_native_instruction_count=len(native_projections['scalar']),
    ), indent=2) + '\n')
    print('PASS exact source/native expansion, aligned pairs/shared scale selectors,29 stage-count scale/bank histories per variant')


if __name__ == '__main__':
    main()
