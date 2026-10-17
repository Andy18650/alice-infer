#!/usr/bin/env python3
"""Verify every delay interval in executable-extracted ISA, not just totals."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'temp/fp8-w8a16/nop-sweep')
isa = (folder / 'nop-sweep.isa').read_text()
meta = (folder / 'metadata.txt').read_text()
summary = {'sha256': hashlib.sha256((folder / 'nop-sweep.hsaco').read_bytes()).hexdigest(), 'kernels': []}
for section in re.split(r'(?=^0[0-9a-f]+ <)', isa, flags=re.M):
    match = re.search(r'<void sweep_kernel<(-?\d+), (\d+)>', section[:300])
    if not match:
        continue
    delay, unroll = map(int, match.groups())
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
        if target < pc and sum('v_mfma_f32_32x32x8f16' in line for line in body) == unroll:
            candidates.append(body)
    assert candidates, (delay, unroll)
    body = min(candidates, key=len)
    counts = Counter(line.strip().split()[0] for line in body)
    # Check each conversion+NOP+MFMA sequence individually. Scalar loop-control
    # instructions may share an interval; no unexpected vector op is allowed.
    sequence = ['v_perm_b32'] * 2 + ['v_pk_ashrrev_i16'] * 2 + ['v_and_b32_e32'] * 2 + ['v_pk_mul_f16'] * 2
    vector_ops, waits, intervals = [], 0, []
    for line in body:
        op = line.strip().split()[0]
        if op == 's_nop':
            waits += int(re.search(r's_nop\s+(\d+)', line)[1]) + 1
        elif op == 'v_mfma_f32_32x32x8f16':
            assert vector_ops == (sequence if delay >= 0 else []), (delay, unroll, vector_ops)
            assert waits == max(delay, 0), (delay, unroll, waits)
            intervals.append(waits)
            vector_ops, waits = [], 0
        elif op.startswith('v_'):
            vector_ops.append(op)
        else:
            assert op in ('s_waitcnt', 's_add_i32', 's_cmp_lt_i32', 's_cbranch_scc1'), (delay, unroll, op)
    assert len(intervals) == unroll and not vector_ops and waits == 0
    assert not any('load_' in line or 'store_' in line for line in body)
    encoded_delay = 'Lin1E' if delay == -1 else f'Li{delay}E'
    block = next(b for b in re.split(r'(?=  - \.agpr_count:)', meta)
                 if re.search(r'\.name:\s*_Z12sweep_kernelI' + encoded_delay + f'Li{unroll}E', b))
    fields = {name: int(re.search(r'\.' + name + r':\s*(\d+)', block)[1]) for name in
              ['agpr_count', 'vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'group_segment_fixed_size']}
    assert fields['private_segment_fixed_size'] == 0 and fields['group_segment_fixed_size'] == 0
    (folder / f'loop-delay{delay}-u{unroll}.txt').write_text('\n'.join(body) + '\n')
    summary['kernels'].append({'delay': delay, 'unroll': unroll, 'instructions': dict(counts), 'resources': fields})
assert len(summary['kernels']) == 135, len(summary['kernels'])
# Independent calibration: 32 scalar NOP delays per backedge, no VALU/MFMA.
clock_isa = (folder / 'nop-clock.isa').read_text()
clock_checked = []
for section in re.split(r'(?=^0[0-9a-f]+ <)', clock_isa, flags=re.M):
    match = re.search(r'<void nop_clock<(\d+)>', section[:200])
    if not match:
        continue
    wait = int(match[1])
    nops = re.findall(r'^\s*s_nop\s+(\d+)', section, flags=re.M)
    assert len(nops) == 32 * ((wait + 15) // 16), (wait, len(nops))
    assert sum(int(n) + 1 for n in nops) == 32 * wait
    clock_body = section.split('s_memtime')[1]
    assert not re.search(r'^\s*v_\w+', clock_body, flags=re.M)
    clock_checked.append(wait)
assert len(clock_checked) == 12
summary['calibration_waits_checked'] = clock_checked
(folder / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS: all 135 sweep kernels; every conversion/delay/MFMA interval exact; no hot-loop memory, scratch or LDS; 12 independent NOP calibration kernels checked')
