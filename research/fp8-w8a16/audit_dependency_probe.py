#!/usr/bin/env python3
"""Audit ring liveness/distance and each staged work group in actual ISA."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'temp/fp8-w8a16/dependency')
isa = (folder / 'dependency-probe.isa').read_text()
meta = (folder / 'metadata.txt').read_text()
summary = {'sha256': hashlib.sha256((folder / 'dependency-probe.hsaco').read_bytes()).hexdigest(), 'kernels': []}
conv = ['v_perm_b32'] * 2 + ['v_pk_ashrrev_i16'] * 2 + ['v_and_b32'] * 2 + ['v_pk_mul_f16'] * 2
mfma = 'v_mfma_f32_32x32x8f16'
def opcode(line):
    return re.sub(r'_e(?:32|64)$', '', line.strip().split()[0])
def dest(line):
    return int(re.search(r'\bv(\d+),', line)[1])
def sources(line):
    return [tuple(map(int, v)) for v in re.findall(r'\bv\[(\d+):(\d+)\]', line)]
for section in re.split(r'(?=^0[0-9a-f]+ <)', isa, flags=re.M):
    match = re.search(r'<void dependency_kernel<(\d+), (\d+), (\d+), (\d+), (\d+)>', section[:350])
    if not match:
        continue
    kind, order, banks, ops, unroll = map(int, match.groups())
    key = (kind, order, banks, ops, unroll)
    instructions = []
    for line in section.splitlines():
        m = re.search(r'// ([0-9A-Fa-f]+):', line)
        if m:
            instructions.append((int(m[1], 16), line))
    candidates = []
    for pc, line in instructions:
        m = re.search(r'\bs_cbranch_\w+\s+(-?\d+)', line)
        if not m:
            continue
        off = int(m[1]); off = off - 65536 if off > 32767 else off
        target = pc + 4 + off * 4
        body = [line for address, line in instructions if target <= address <= pc]
        if target < pc and sum(mfma in line for line in body) == unroll:
            candidates.append(body)
    assert candidates, key
    body = min(candidates, key=len)
    vector = [line for line in body if opcode(line).startswith('v_')]
    workop = 'v_nop' if kind == 2 else 'v_or_b32' if kind in (5, 7) else 'v_xor_b32'
    work = [] if kind in (0, 1) else conv if kind == 9 else [workop] * ops
    seq = (work + [mfma]) if order == 0 else ([mfma] + work)
    # Partial source writes can make LLVM insert physical bank-copy MOVs.
    # Retain/count these explicitly; do not mislabel them as requested work.
    copies = [line for line in vector if opcode(line) == 'v_mov_b32']
    semantic = [line for line in vector if opcode(line) != 'v_mov_b32']
    assert [opcode(line) for line in semantic] == seq * unroll, key
    if copies:
        assert kind in (5, 7, 8, 10), (key, copies)
    assert not any('load_' in line or 'store_' in line for line in body), key
    nops = [int(re.search(r's_nop\s+(\d+)', line)[1]) for line in body if opcode(line) == 's_nop']
    if kind == 10:
        assert nops.count(3) >= unroll, (key, nops)
    elif kind not in (5, 7, 8, 9):
        assert not nops, (key, nops)
    # Check actual A/B register reads and writes, not only opcode categories.
    width = len(seq)
    for j in range(unroll):
        unit = semantic[j * width:(j + 1) * width]
        matrix = unit[-1] if order == 0 else unit[0]
        group = unit[:-1] if order == 0 else unit[1:]
        if order == 1 and kind in (5, 7, 8, 10):
            # Written values feed the FOLLOWING MFMA, potentially via copies.
            matrix = semantic[((j + 1) % unroll) * width]
        ar, br = sources(matrix)
        aset, bset = set(range(ar[0], ar[1] + 1)), set(range(br[0], br[1] + 1))
        if kind in (4, 6):
            expected = bset if kind == 4 else aset
            for line in group:
                regs = list(map(int, re.findall(r'\bv(\d+)\b', line.split('//')[0])))
                assert regs[-1] in expected and regs[0] not in aset | bset, (key, line)
        if kind in (5, 7):
            expected = bset if kind == 5 else aset
            assert all(dest(line) in expected for line in group), key
        if kind in (8, 10):
            assert all(dest(line) in (aset if k % 2 == 0 else bset) for k, line in enumerate(group)), key
        if kind == 3:
            assert all(dest(line) not in aset | bset for line in group), key
    if kind == 9:
        # Two laps provide a previous producer for every consumed physical bank.
        last_produced = {}; counter = 0; distances = []
        for lap in range(2):
            produced = []
            for line in vector:
                op = opcode(line)
                if op == 'v_pk_mul_f16':
                    produced.append(dest(line))
                    if len(produced) == 2:
                        last_produced[tuple(produced)] = counter; produced = []
                elif op == mfma:
                    lo, hi = sources(line)[1]
                    if lap == 1:
                        distances.append(counter - last_produced[tuple(range(lo, hi + 1))])
                    counter += 1
        assert set(distances) == {banks - 1}, (key, distances)
        assert len({sources(line)[1] for line in vector if opcode(line) == mfma}) == banks, key
        for op in ('v_perm_b32', 'v_pk_ashrrev_i16', 'v_pk_mul_f16'):
            assert sum(opcode(line) == op for _, line in instructions) == 2 * (unroll + banks), (key, op)
    scalars = []; nm = nv = 0
    for line in body:
        op = opcode(line)
        if op == mfma:
            nm += 1
        elif op.startswith('v_'):
            nv += 1
        else:
            assert op in ('s_waitcnt', 's_add_i32', 's_cmp_lt_i32', 's_cbranch_scc1', 's_nop'), (key, op)
            scalars.append({'instruction': line.strip().split('//')[0].strip(), 'preceding_mfmas': nm, 'preceding_valu': nv})
    suffix = ''.join(f'Li{n}E' for n in key)
    block = next(b for b in re.split(r'(?=  - \.agpr_count:)', meta)
                 if re.search(r'\.name:\s*_Z17dependency_kernelI' + suffix + 'E', b))
    fields = {name: int(re.search(r'\.' + name + r':\s*(\d+)', block)[1]) for name in
              ['agpr_count', 'vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'group_segment_fixed_size']}
    assert fields['private_segment_fixed_size'] == 0 and fields['group_segment_fixed_size'] == 0, key
    (folder / ('loop-' + '-'.join(map(str, key)) + '.txt')).write_text('\n'.join(body) + '\n')
    summary['kernels'].append(dict(zip(('kind', 'order', 'banks', 'ops', 'unroll'), key)) | {'resources': fields, 'scalars': scalars, 'copy_movs_per_mfma': len(copies) / unroll, 'nop_wait_units_per_loop': sum(n + 1 for n in nops), 'instructions': dict(Counter(opcode(line) for line in body))})
assert len(summary['kernels']) == 309, len(summary['kernels'])
(folder / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS 309 kernels: work/MFMA order, physical source checks, 1/3-intervening-MFMA ring distances, zero private/LDS bytes; extra copy MOVs and scalar positions recorded explicitly')
