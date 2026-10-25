#!/usr/bin/env python3
"""Audit extracted native instructions and the no-spill resource contract.

The full timed body includes startup and two alternative tails. The hot loop
is the backward branch spanning the two last-fragment-parity phases. Counts
are deliberately checked for both regions; source text alone is not evidence
that the expected instructions made it into the executable.
"""

import argparse
import collections
import json
import re
from pathlib import Path


MFMA = 'v_mfma_f32_32x32x8f16'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_instructions(disassembly):
    instructions = []
    for line in disassembly.splitlines():
        match = re.search(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):', line)
        if match:
            instructions.append((int(match[2], 16), match[1]))
    return instructions


def opcode_counts(instructions):
    return collections.Counter(instruction.split()[0] for instruction in instructions)


def integer_field(text, pattern):
    match = re.search(pattern, text)
    require(match is not None, f'Missing field: {pattern}')
    return int(match[1])


def find_hot_loop(instructions, expected_mfma_count):
    candidates = []
    for index, (pc, instruction) in enumerate(instructions):
        if not instruction.startswith(('s_branch ', 's_cbranch_scc1 ')):
            continue

        # SOPP branches use a signed 16-bit displacement in four-byte units,
        # relative to the instruction after the branch.
        displacement = int(instruction.split()[1], 0)
        if displacement >= 32768:
            displacement -= 65536
        if displacement >= 0:
            continue
        target = pc + 4 + 4 * displacement
        body = [text for address, text in instructions[:index + 1] if address >= target]
        if opcode_counts(body)[MFMA] == expected_mfma_count:
            candidates.append(body)

    require(len(candidates) == 1, f'Expected one hot loop, found {len(candidates)}')
    return candidates[0]


def audit(output_directory):
    header = (output_directory / 'compute.hpp').read_text()
    tile = integer_field(header, r'kTile=(\d+)')
    stage_k = integer_field(header, r'kStageK=(\d+)')
    micro_tiles = tile // 64
    pairs = micro_tiles // 2
    fragments = stage_k // 8
    resident = 'kResident=true' in header
    fp8_b = 'kFp8BStorage=true' in header
    schedule_match = re.search(r'kFp8Schedule=(\d+)', header)
    fp8_schedule = int(schedule_match[1]) if schedule_match else 0

    disassembly = (output_directory / 'probe.isa').read_text()
    instructions = parse_instructions(disassembly)
    start = next(i for i, (_, text) in enumerate(instructions) if text.startswith('s_memtime'))
    end = next(
        i for i in range(start + 1, len(instructions))
        if instructions[i][1].startswith('s_memtime')
    )
    timed_body = [text for _, text in instructions[start:end + 1]]
    counts = opcode_counts(timed_body)
    expected_body_counts = {
        MFMA: (1 if resident else 8) * fragments * micro_tiles * micro_tiles,
        'global_load_dwordx4': 15 if fp8_b else (2 if resident else 5) * fragments * pairs,
        'ds_write_b128': (1 if resident else 6) * fragments * pairs,
        'ds_read_b128': (2 if resident else 12) * fragments * pairs,
    }
    for opcode, expected in expected_body_counts.items():
        require(counts[opcode] == expected,
                f'Timed body {opcode}: expected {expected}, found {counts[opcode]}')
    require(
        not any(re.search(r'accvgpr|scratch_|readlane|writelane|v_cvt', text)
                for text in timed_body),
        'Unexpected accumulator transfer, conversion or spill inside timing',
    )

    notes = (output_directory / 'metadata.txt').read_text()
    private_bytes = integer_field(notes, r'\.private_segment_fixed_size:\s*(\d+)')
    static_lds_bytes = integer_field(notes, r'\.group_segment_fixed_size:\s*(\d+)')
    require(private_bytes == 0, f'Kernel spills to private memory: {private_bytes} bytes')
    require(static_lds_bytes == 0, 'Expected dynamic LDS only')
    require(not re.search(r'scratch_|(?:read|write)lane', disassembly),
            'Scratch or lane-spill instruction present')

    expected_loop_mfmas = (1 if resident else 2) * fragments * micro_tiles * micro_tiles
    hot_loop = find_hot_loop(instructions, expected_loop_mfmas)
    loop_counts = opcode_counts(hot_loop)
    expected_loop_counts = {
        's_barrier': 4,
        'global_load_dwordx4': 6 if fp8_b else 2 * fragments * pairs,
        'ds_read_b128': 4 * fragments * pairs,
        'ds_write_b128': 2 * fragments * pairs,
    }
    for opcode, normal_count in expected_loop_counts.items():
        expected = 0 if resident else normal_count
        require(loop_counts[opcode] == expected,
                f'Hot loop {opcode}: expected {expected}, found {loop_counts[opcode]}')
    require(not any('s_nop' in text for text in hot_loop), 'NOP in hot loop')
    if fp8_b:
        for opcode in ('v_perm_b32', 'v_pk_ashrrev_i16', 'v_and_b32_e32', 'v_pk_mul_f16'):
            require(loop_counts[opcode] == 16 and counts[opcode] == 48,
                    f'Wrong FP8 conversion count for {opcode}')
        require(len(hot_loop) == 147, 'Unexpected balanced FP8 loop length')
        if fp8_schedule:
            for index, instruction in enumerate(hot_loop):
                if instruction != 's_waitcnt vmcnt(0)':
                    continue
                require(all(op.startswith('v_perm_b32')
                            for op in hot_loop[index + 4:index + 12]),
                        'Expected eight clustered source-copy permutations')
                require(hot_loop[index + 12].startswith('global_load_dwordx4 v[60:63]'),
                        'B prefetch must immediately follow the last permutation')
                if fp8_schedule == 2:
                    require(hot_loop[index + 21].startswith(MFMA),
                            'Expected current q0/C01 after six conversion arithmetic ops')
    if not resident:
        # Two stages, one bottom-controlled backedge. Boundary tests belong
        # only to the bounded tails, not either steady phase.
        expected_control = {
            's_cmp_eq_u32': 0,
            's_cmp_le_u32': 0,
            's_sub_u32': 1,
            's_cmp_ge_u32': 1,
            's_cbranch_scc1': 1,
            's_branch': 0,
        }
        for opcode, expected in expected_control.items():
            require(loop_counts[opcode] == expected,
                    f'Unexpected steady-loop control: {opcode}={loop_counts[opcode]}')
        for index, instruction in enumerate(hot_loop):
            if instruction == 's_waitcnt vmcnt(0)':
                require(hot_loop[index + 1].startswith(MFMA),
                        'Expected one current-q0 MFMA immediately after head VM0')
                require(hot_loop[index + 2].startswith('ds_write_b128'),
                        'Expected LDS stores immediately after the moved MFMA')
    require(not any(text.startswith('s_branch ') and int(text.split()[1], 0) == 0
                    for text in timed_body), 'Redundant fall-through branch in timed body')

    (output_directory / 'body.txt').write_text('\n'.join(timed_body) + '\n')
    (output_directory / 'loop.txt').write_text('\n'.join(hot_loop) + '\n')
    inventory = {
        name: integer_field(notes, r'\.' + name + r':\s*(\d+)')
        for name in ('vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'agpr_count')
    }
    inventory.update(
        tile=tile,
        stage_k=stage_k,
        kind='resident-control' if resident else 'gemm',
        storage='fp16-a/fp8-b' if fp8_b else 'fp16-a/fp16-b',
        fp8_schedule=fp8_schedule if fp8_b else None,
        lds_bytes=tile * stage_k * 4,
        instructions_per_loop=dict(loop_counts),
    )
    (output_directory / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    audit(args.output_directory)


if __name__ == '__main__':
    main()
