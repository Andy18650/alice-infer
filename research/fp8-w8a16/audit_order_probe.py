#!/usr/bin/env python3
"""Check exact MFMA/conversion phase and prologue in the executable object."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'temp/fp8-w8a16/order')
isa = (folder / 'order-probe.isa').read_text()
meta = (folder / 'metadata.txt').read_text()
summary = {'sha256': hashlib.sha256((folder / 'order-probe.hsaco').read_bytes()).hexdigest(), 'kernels': []}
conversion = ['v_perm_b32'] * 2 + ['v_pk_ashrrev_i16'] * 2 + ['v_and_b32_e32'] * 2 + ['v_pk_mul_f16'] * 2
mfma = 'v_mfma_f32_32x32x8f16'
for section in re.split(r'(?=^0[0-9a-f]+ <)', isa, flags=re.M):
    match = re.search(r'<void order_kernel<(\d+), (\d+)>', section[:300])
    if not match:
        continue
    mode, unroll = map(int, match.groups())
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
        offset = int(m[1])
        if offset > 32767:
            offset -= 65536
        target = pc + 4 + offset * 4
        body = [line for address, line in instructions if target <= address <= pc]
        if target < pc and sum(mfma in line for line in body) == unroll:
            candidates.append(body)
    assert candidates, (mode, unroll)
    body = min(candidates, key=len)
    opcodes = [line.strip().split()[0] for line in body]
    vector_ops = [op for op in opcodes if op.startswith('v_')]
    sequence = [mfma] if mode == 0 else conversion + [mfma] if mode == 1 else [mfma] + conversion
    assert vector_ops == sequence * unroll, (mode, unroll, vector_ops)
    assert not any('load_' in op or 'store_' in op or op == 's_nop' for op in opcodes)
    assert all(op.startswith('v_') or op in ('s_waitcnt', 's_add_i32', 's_cmp_lt_i32', 's_cbranch_scc1') for op in opcodes)
    # Count the two additional conversions outside the mode-2 hot loop.
    full_ops = Counter(line.strip().split()[0] for _, line in instructions)
    for op in ('v_perm_b32', 'v_pk_ashrrev_i16', 'v_pk_mul_f16'):
        assert full_ops[op] == (0 if mode == 0 else 2 * unroll + (4 if mode == 2 else 0)), (mode, unroll, op)
    block = next(b for b in re.split(r'(?=  - \.agpr_count:)', meta)
                 if re.search(r'\.name:\s*_Z12order_kernelILi' + str(mode) + r'ELi' + str(unroll) + r'EE', b))
    fields = {name: int(re.search(r'\.' + name + r':\s*(\d+)', block)[1]) for name in
              ['agpr_count', 'vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'group_segment_fixed_size']}
    assert fields['private_segment_fixed_size'] == 0 and fields['group_segment_fixed_size'] == 0
    (folder / f'loop-mode{mode}-u{unroll}.txt').write_text('\n'.join(body) + '\n')
    summary['kernels'].append({'mode': mode, 'unroll': unroll, 'instructions': dict(Counter(opcodes)), 'head': body[0].strip(), 'resources': fields})
assert len(summary['kernels']) == 9
(folder / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS all nine entries: exact before/after order, two prologue conversions for after, no loop NOPs/memory, no private/LDS bytes')
