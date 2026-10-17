#!/usr/bin/env python3
"""Audit the actual executable-extracted gfx90a object, not compiler text."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'temp/fp8-w8a16')
isa = (folder / 'probe.isa').read_text()
meta = (folder / 'metadata.txt').read_text()
summary = {'object_sha256': hashlib.sha256((folder / 'probe.hsaco').read_bytes()).hexdigest(), 'kernels': {}}
for mode in range(4):
    section = next(s for s in re.split(r'(?=^0[0-9a-f]+ <)', isa, flags=re.M)
                   if f'<void throughput<{mode}>' in s[:350])
    instructions = []
    for line in section.splitlines():
        match = re.search(r'// ([0-9A-Fa-f]+):', line)
        if match:
            instructions.append((int(match[1], 16), line))
    candidates = []
    for pc, line in instructions:
        match = re.search(r'\bs_cbranch_\w+\s+(-?\d+)', line)
        if not match:
            continue
        offset = int(match[1])
        if offset > 32767:
            offset -= 65536
        target = pc + 4 + offset * 4
        if target >= pc:
            continue
        body = [line for address, line in instructions if target <= address <= pc]
        if sum('v_mfma_f32_32x32x8f16' in line for line in body) == 8:
            candidates.append(body)
    assert candidates, mode
    body = min(candidates, key=len)
    counts = Counter(line.strip().split()[0] for line in body)
    assert counts['v_mfma_f32_32x32x8f16'] == 8
    for opcode in ['v_perm_b32', 'v_pk_ashrrev_i16', 'v_and_b32_e32', 'v_pk_mul_f16']:
        assert counts[opcode] == (16 if mode in (1, 3) else 0), (mode, opcode, counts)
    loads = sum(count for op, count in counts.items() if 'load_' in op)
    assert loads == (8 if mode >= 2 else 0), (mode, counts)
    if mode == 1:
        assert not any('s_nop' in line for line in body)
    # Metadata, rather than absence of a scratch mnemonic, establishes no spill.
    block = next(b for b in re.split(r'(?=  - \.agpr_count:)', meta)
                 if re.search(r'\.name:\s*_Z10throughputILi' + str(mode) + r'EE', b))
    fields = {name: int(re.search(r'\.' + name + r':\s*(\d+)', block)[1]) for name in
              ['agpr_count', 'vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'group_segment_fixed_size']}
    assert fields['private_segment_fixed_size'] == 0
    assert fields['group_segment_fixed_size'] == 0
    (folder / f'hot-loop-{mode}.txt').write_text('\n'.join(body) + '\n')
    summary['kernels'][mode] = {'instructions': dict(counts), 'resources': fields}
(folder / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
