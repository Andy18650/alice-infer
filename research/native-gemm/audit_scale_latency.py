#!/usr/bin/env python3
"""Check SMEM readiness, scale-queue ownership, bounded padding and read moves.

Pending scalar loads become visible ONLY at an explicit LGKM0. This is an
order/ownership model, not a cache-latency or hardware-performance emulator.
"""

import argparse
import collections
import json
import re
from pathlib import Path

from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, require
from audit_scaled_packed import expand
from inspect_code import parse_instructions


def restore_late_read_order(code):
    result = list(code)
    moves = 0
    index = 1
    while index + 2 < len(result):
        if (result[index].startswith('ds_read_b128 v[8:11]')
                and result[index + 1].startswith('ds_read_b128 v[12:15]')
                and result[index - 1].startswith('v_mfma_')
                and result[index + 2].startswith(('v_fma_f32', 'v_pk_fma_f32'))):
            reads = result[index:index + 2]
            del result[index:index + 2]
            end = next(i for i in range(index, len(result)) if result[i].startswith('v_mfma_'))
            result[end:end] = reads
            moves += 1
            index = end + 2
        else:
            index += 1
    return result, moves


def trace(code, stages, batch):
    labels = {op[:-1]: i for i, op in enumerate(code) if op.endswith(':')}
    remaining, pc, steps, mfmas, requests, next_scale = stages, 0, 0, 0, 0, 0
    groups = (stages + 3) // 4
    scalar = {14: None, 15: None, 20: None, 21: None}
    pending = {}
    pending_start = None
    lead = []
    raw = {register: (None, 0) for register in range(128, 256)}
    outputs = {register: [] for register in range(64, 128)}
    counts = collections.Counter()
    scc = False
    executed = []
    while pc < len(code):
        steps += 1
        require(steps < 2000000, 'Nonterminating control')
        op = code[pc]
        pc += 1
        if op.endswith(':'): continue
        if op == 's_memtime s[6:7]': break
        opcode = op.split()[0]
        if opcode.startswith('s_cmp_'):
            require('s16,' in op, 'Unexpected comparison')
            rhs = int(op.rsplit(',', 1)[1])
            scc = {'s_cmp_eq_u32': remaining == rhs, 's_cmp_le_u32': remaining <= rhs,
                   's_cmp_ge_u32': remaining >= rhs}[opcode]
        elif opcode == 's_cbranch_scc1':
            if scc: pc = labels[op.split()[1]]
        elif opcode == 's_branch': pc = labels[op.split()[1]]
        elif op.startswith('s_sub_u32 s16'):
            remaining -= int(op.rsplit(',', 1)[1])
            require(remaining >= 0, 'Counter underflow')
        elif op in ('s_mov_b32 s15, s14', 's_mov_b32 s14, s20', 's_mov_b32 s14, s21'):
            destination, source = map(int, re.findall(r'\bs(\d+)\b', op))
            require(source not in pending and destination not in pending, 'Read/overwrite pending SMEM result')
            scalar[destination] = scalar[source]
        elif op.startswith(('s_load_dword s14,', 's_load_dwordx2 s[20:21],')):
            require(not pending, 'Unexpected overlapping scalar requests')
            destinations = (20, 21) if batch else (14,)
            for register in destinations:
                pending[register] = next_scale if next_scale < groups else None
                next_scale += 1
            requests += 1
            pending_start = mfmas
        elif op == 's_waitcnt lgkmcnt(0)':
            if pending:
                lead.append(mfmas - pending_start)
                scalar.update(pending)
                pending.clear()
        elif opcode.startswith('v_mfma_'):
            first, last = map(int, re.match(r'v_mfma_\S+ v\[(\d+):(\d+)\]', op).groups())
            group = mfmas // 64
            for register in range(first, last + 1):
                owner, contributions = raw[register]
                if op.endswith(', 0'): owner, contributions = group, 0
                require(owner == group, 'Raw bank not reset')
                raw[register] = owner, contributions + 1
            mfmas += 1
        elif opcode == 'v_fma_f32':
            destination, source, scale, accumulator = map(int, re.fullmatch(
                r'v_fma_f32 v(\d+), v(\d+), s(\d+), v(\d+)', op).groups())
            require(scale not in pending, 'Scaling consumes an uncompleted scalar load')
            owner, contributions = raw[source]
            require(destination == accumulator and owner == scalar[scale], 'Scale/raw ownership mismatch')
            if owner is not None:
                require(contributions == min(4, stages - owner * 4) * 4, 'Incomplete K128 partial')
                require(outputs[destination] == list(range(owner)), 'Missing/reordered scale group')
                outputs[destination].append(owner)
        counts[opcode] += 1
        if not opcode.startswith(('s_cmp_', 's_cbranch', 's_branch')):
            executed.append(op)
    require(not pending, 'Uncompleted metadata at final flush')
    require(all(history == list(range(groups)) for history in outputs.values()), 'Incorrect final sums')
    require(mfmas == 16 * stages and counts['global_load_dwordx4'] == 3 * stages,
            'MFMA/weight overfetch')
    require(counts['ds_write_b128'] == 4 * stages and counts['ds_read_b128'] == 8 * stages,
            'LDS payload changed')
    expected_requests = (groups + 1) // 2 if batch else groups
    require(requests == expected_requests, 'Wrong metadata request count')
    require(next_scale == (groups + 1) // 2 * 2 if batch else next_scale == groups,
            'Metadata access beyond explicit pair padding')
    require(all(distance >= 8 for distance in lead), 'No q2/q3 MFMA lead before metadata wait')
    return dict(stages=stages, groups=groups, requests=requests,
                physical_scale_words=next_scale, minimum_mfmas_before_wait=min(lead) if lead else None), executed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/scale-latency-audit')
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists() or not any(out.iterdir()), 'Choose a fresh output')
    out.mkdir(parents=True, exist_ok=True)
    proofs = {}
    flows = {}
    for plan in ('late', 'batch', 'read'):
        build = ROOT / f'temp/native-gemm/scale-latency/{plan}'
        code = [item for op in asm_instructions((build / 'compute.hpp').read_text()) for item in expand(op)]
        results = [trace(code, count, plan == 'batch') for count in STAGE_COUNTS]
        proofs[plan] = [result[0] for result in results]
        flows[plan] = [result[1] for result in results]
    for index, stages in enumerate(STAGE_COUNTS):
        restored, moves = restore_late_read_order(flows['read'][index])
        require(restored == flows['late'][index], 'Read experiment changes more than q1 read positions')
        require(moves == (stages + 2) // 4, 'Early reads issued in a final/non-boundary stage')
    from inspect_code import find_hot_loop
    baseline = parse_instructions((ROOT / 'temp/native-gemm/scaled-packed-fixed/scaled-all/probe.isa').read_text())
    late = parse_instructions((ROOT / 'temp/native-gemm/scale-latency/late/probe.isa').read_text())
    def without_scale_request(loop):
        return [op for op in loop if not op.startswith((
            's_load_dword s14,', 's_add_u32 s12,', 's_addc_u32 s13,'))]
    require(without_scale_request(find_hot_loop(baseline, 128))
            == without_scale_request(find_hot_loop(late, 128)),
            'Late experiment changed non-metadata hot-loop instructions')
    read = parse_instructions((ROOT / 'temp/native-gemm/scale-latency/read/probe.isa').read_text())
    restored, moves = restore_late_read_order([op for _, op in read])
    require(restored == [op for _, op in late], 'Native read movement not isolated')
    (out / 'proof.json').write_text(json.dumps(dict(
        readiness_and_bank_checks=proofs, native_read_moves=moves,
        native_read_projection_identical=True, late_only_retimes_metadata_hot_loop=True,
    ), indent=2) + '\n')
    print('PASS29 stage counts/variant: explicit metadata readiness,ordered scales/banks,pair padding,no overfetch;native read movement isolated')


if __name__ == '__main__':
    main()
