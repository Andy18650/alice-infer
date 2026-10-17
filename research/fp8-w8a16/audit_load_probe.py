#!/usr/bin/env python3
"""Verify real per-step loads and immediate waits in the untuned baseline."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'temp/fp8-w8a16/load')
isa = (folder / 'load-probe.isa').read_text()
meta = (folder / 'metadata.txt').read_text()
mfma = 'v_mfma_f32_32x32x8f16'
conversion = ['v_perm_b32'] * 2 + ['v_pk_ashrrev_i16'] * 2 + ['v_and_b32'] * 2 + ['v_pk_mul_f16'] * 2
summary = {'sha256': hashlib.sha256((folder / 'load-probe.hsaco').read_bytes()).hexdigest(), 'kernels': []}
def op(line): return re.sub(r'_e(?:32|64)$', '', line.strip().split()[0])
for section in re.split(r'(?=^0[0-9a-f]+ <)', isa, flags=re.M):
    match = re.search(r'<void load_kernel<(\d+), (\d+)>', section[:300])
    if not match: continue
    mode, unroll = map(int, match.groups()); key = mode, unroll
    ins = []
    for line in section.splitlines():
        m = re.search(r'// ([0-9A-Fa-f]+):', line)
        if m: ins.append((int(m[1], 16), line))
    candidates = []
    for pc, line in ins:
        m = re.search(r'\bs_cbranch_\w+\s+(-?\d+)', line)
        if not m: continue
        off = int(m[1]); off = off - 65536 if off > 32767 else off
        start = pc + 4 + off * 4
        body = [l for p, l in ins if start <= p <= pc]
        if start < pc and sum(mfma in l for l in body) == unroll: candidates.append((start, body))
    assert candidates, key
    start, body = min(candidates, key=lambda x: len(x[1]))
    counts = Counter(op(line) for line in body)
    loads = [line for line in body if op(line).startswith('global_load_')]
    assert len(loads) == (unroll if mode >= 2 else 0), key
    for index, line in enumerate(body):
        if op(line).startswith('global_load_'):
            assert op(line) == ('global_load_dwordx2' if mode == 2 else 'global_load_dword'), key
            assert 's_waitcnt vmcnt(0)' in body[index + 1], (key, line)
    if mode >= 2:
        # This deliberately proves full draining, not overlap/prefetch quality.
        assert sum('s_waitcnt vmcnt(0)' in line for line in body) >= unroll, key
    for opcode in ('v_perm_b32', 'v_pk_ashrrev_i16', 'v_pk_mul_f16'):
        assert counts[opcode] == (2 * unroll if mode in (1, 3) else 0), (key, opcode)
    # Address work stays in the totals. An unexpected extra AND would fail
    # this conversion-pattern check rather than being silently ignored.
    semantic = [op(line) for line in body if op(line) in set(conversion + [mfma]) or op(line).startswith('global_load_')]
    expected = [mfma]
    if mode == 2: expected += ['global_load_dwordx2']
    if mode == 3: expected += ['global_load_dword']
    if mode in (1, 3): expected += conversion
    assert semantic == expected * unroll, key
    assert not any('store_' in op(line) or op(line).startswith('ds_') or op(line) in ('v_readlane_b32', 'v_writelane_b32') for line in body), key
    block = next(b for b in re.split(r'(?=  - \.agpr_count:)', meta)
                 if re.search(r'\.name:\s*_Z11load_kernelILi' + str(mode) + 'ELi' + str(unroll) + 'EE', b))
    fields = {name: int(re.search(r'\.' + name + r':\s*(\d+)', block)[1]) for name in
              ['agpr_count', 'vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'group_segment_fixed_size']}
    assert fields['private_segment_fixed_size'] == 0 and fields['group_segment_fixed_size'] == 0, key
    address_ops = {opcode: count for opcode, count in counts.items()
                   if opcode.startswith('v_') and opcode not in set(conversion + [mfma])}
    (folder / f'loop-mode{mode}-u{unroll}.txt').write_text('\n'.join(body) + '\n')
    summary['kernels'].append({'mode': mode, 'unroll': unroll, 'resources': fields, 'instructions': dict(counts), 'other_vector_ops': address_ops})
assert len(summary['kernels']) == 12
(folder / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS 12 entries: exact MFMA/load/conversion counts, every global load immediately followed by vmcnt(0), no private/LDS or lane spills; address ops recorded')
