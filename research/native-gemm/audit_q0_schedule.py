#!/usr/bin/env python3
"""Check early-load staging lifetimes, pointer streams and fixed math/fences.

Global loads are pending until VM0. The model rejects staging overwrite before
its LDS consumer, and B reload before all permutation copies. It does not
simulate GPU instruction latency; actual numerical and ISA tests are separate.
"""

import argparse
import collections
import json
from pathlib import Path

from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, require
from audit_scale_latency import trace
from audit_scaled_packed import expand
from inspect_code import parse_instructions


A_LOAD = ('global_load_dwordx4 v[44:47]', 'global_load_dwordx4 v[48:51]')


def restore_a_loads(code):
    result = list(code)
    heads = [i for i, op in enumerate(result) if op == 's_waitcnt vmcnt(0)'
             and i + 1 < len(result) and result[i + 1].startswith('v_mfma_')]
    moves = 0
    for head in heads:
        end = next(i for i in range(head + 1, len(result)) if result[i] == 's_waitcnt lgkmcnt(0)')
        loads = [i for i in range(head + 1, end) if result[i].startswith(A_LOAD)]
        if not loads: continue
        require(len(loads) == 2, 'Expected one A packet pair')
        packet = [result[i] for i in loads]
        for i in reversed(loads): result.pop(i)
        target = next(i for i in range(head + 1, len(result)) if result[i].startswith('s_add_u32 s0,'))
        result[target:target] = packet
        moves += 1
    return result, moves


def staging(trace_ops, stages):
    pointer = {'a': 0, 'b': 0}
    owner = {44: None, 48: None, 60: None}
    consumed = {44: True, 48: True, 60: True}
    pending = {}
    copied = set()
    expanded = {}
    progress = {}
    producer = 0
    for op in trace_ops:
        if op.startswith('global_load_dwordx4'):
            register = int(op.split('v[')[1].split(':')[0])
            kind = 'b' if register == 60 else 'a'
            require(register not in pending and consumed[register], 'Reload overwrites unconsumed staging')
            pending[register] = pointer[kind]
        elif op == 's_waitcnt vmcnt(0)':
            for register, value in pending.items():
                owner[register] = value
                consumed[register] = False
                if register == 60: copied.clear()
            pending.clear()
        elif op.startswith('s_add_u32 s0,'): pointer['a'] += 1
        elif op.startswith('s_add_u32 s2,'): pointer['b'] += 1
        elif op.startswith('v_perm_b32'):
            destination = int(op.split()[1].rstrip(',')[1:])
            require(60 not in pending and owner[60] == producer, 'Copy reads wrong/pending B stage')
            expanded[destination] = producer
            progress[destination] = 0
            copied.add(destination)
            if len(copied) == 8: consumed[60] = True
        elif op.startswith(('v_pk_ashrrev_i16','v_and_b32','v_pk_mul_f16')):
            destination = int(op.split()[1].rstrip(',')[1:])
            require(expanded[destination] == producer, 'Wrong expanded B stage')
            progress[destination] += 1
        elif op.startswith('ds_write_b128'):
            register = int(op.split('v[')[1].split(':')[0])
            if register in (44, 48):
                require(register not in pending and owner[register] == producer
                        and not consumed[register], 'A store reads wrong/overwritten/pending packet')
                consumed[register] = True
            else:
                require(register in (52, 56), 'Unknown producer packet')
                require(all(expanded[r] == producer and progress[r] == 3
                            for r in range(register, register + 4)), 'B store before full conversion')
                if register == 56: producer += 1
    require(not pending and producer == stages, 'Incomplete staging stream')
    require(pointer == {'a': stages, 'b': stages}, 'Wrong K source-pointer advances')


def projections(ops):
    categories = {
        'a_stream': lambda op: op.startswith((*A_LOAD, 's_add_u32 s0,', 's_addc_u32 s1,')),
        'b_stream': lambda op: op.startswith(('global_load_dwordx4 v[60:63]', 's_add_u32 s2,', 's_addc_u32 s3,')),
        'stores': lambda op: op.startswith('ds_write_b128'),
        'conversion': lambda op: op.startswith(('v_perm_b32','v_pk_ashrrev_i16','v_and_b32','v_pk_mul_f16')),
        'compute_read_fence_scale': lambda op: op.startswith((
            'v_mfma_', 'v_fma_f32', 'v_pk_fma_f32', 'ds_read_b128', 's_waitcnt', 's_barrier',
            's_load_dword s14', 's_mov_b32 s15,', 's_add_u32 s12,', 's_addc_u32 s13,')),
    }
    return {name: [op for op in ops if keep(op)] for name, keep in categories.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/q0-schedule/checks')
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists() or not any(out.iterdir()), 'Choose a fresh output')
    out.mkdir(parents=True, exist_ok=True)
    builds = {'baseline': ROOT / 'temp/native-gemm/scale-latency/late',
              **{p: ROOT / f'temp/native-gemm/q0-schedule/{p}' for p in ('earlya', 'balanced')}}
    code = {name: [x for op in asm_instructions((build / 'compute.hpp').read_text()) for x in expand(op)]
            for name, build in builds.items()}
    checks = []
    for count in STAGE_COUNTS:
        results = {name: trace(ops, count, False) for name, ops in code.items()}
        for name, (_, executed) in results.items():
            staging(executed, count)
            require(collections.Counter(executed) == collections.Counter(results['baseline'][1]),
                    f'Changed executed instruction multiset: {name}/{count}')
            require(projections(executed) == projections(results['baseline'][1]),
                    f'Changed ordered semantic stream: {name}/{count}')
        restored, moves = restore_a_loads(results['earlya'][1])
        require(restored == results['baseline'][1] and moves == max(0, count - 2),
                'Early-A experiment changes more than two future loads per eligible stage')
        checks.append(dict(stages=count, readiness=results['balanced'][0], early_a_pairs=moves))
    native = {name: [op for _, op in parse_instructions((build / 'probe.isa').read_text())]
              for name, build in builds.items()}
    restored, _ = restore_a_loads(native['earlya'])
    require(restored == native['baseline'], 'Native early-A change not isolated')
    require(projections(native['balanced']) == projections(native['baseline']), 'Native semantic stream changed')
    require(collections.Counter(native['balanced']) == collections.Counter(native['baseline']),
            'Native balanced instruction multiset changed')
    inventories = {name: json.loads((build / 'inventory.json').read_text()) for name, build in builds.items()}
    for name in ('earlya','balanced'):
        for field in ('vgpr_count','agpr_count','sgpr_count','private_segment_fixed_size','lds_bytes','instructions_per_loop'):
            require(inventories[name][field] == inventories['baseline'][field], f'Resource/count change:{name}/{field}')
    (out / 'summary.json').write_text(json.dumps(dict(checks=checks,
        native_early_a_only_load_moves=True,native_balanced_semantic_streams_identical=True,
        same_resources_and_instruction_counts=True),indent=2)+'\n')
    print('PASS29 stage counts: staging lifetimes,pointer streams,conversion,math/scales/fences;native moves/projections/counts/resources')


if __name__ == '__main__': main()
