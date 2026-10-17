#!/usr/bin/env python3
"""Check exact hand-scheduled intervals, preheader placement and register lifetime."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'temp/fp8-w8a16/split')
isa = (folder / 'split-probe.isa').read_text()
meta = (folder / 'metadata.txt').read_text()
mfma = 'v_mfma_f32_32x32x8f16'
conv = ['v_perm_b32'] * 2 + ['v_pk_ashrrev_i16'] * 2 + ['v_and_b32'] * 2 + ['v_pk_mul_f16'] * 2
def op(line): return re.sub(r'_e(?:32|64)$', '', line.strip().split()[0])
def sources(line): return [tuple(map(int, x)) for x in re.findall(r'\bv\[(\d+):(\d+)\]', line)]
summary = {'sha256': hashlib.sha256((folder / 'split-probe.hsaco').read_bytes()).hexdigest(), 'kernels': []}
for section in re.split(r'(?=^0[0-9a-f]+ <)', isa, flags=re.M):
    match = re.search(r'<void split_kernel<(\d+), (\d+), (\d+)>', section[:300])
    if not match: continue
    schedule, prelude, unroll = map(int, match.groups()); key = (schedule, prelude, unroll)
    split = 0 if schedule == 0 else schedule - 6 if schedule >= 6 else schedule - 1
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
        if start < pc and sum(mfma in l for l in body) == unroll:
            candidates.append((start, body))
    assert candidates, key
    start, body = min(candidates, key=lambda x: len(x[1]))
    vec = [line for line in body if op(line).startswith('v_')]
    if schedule == 0: expected = [mfma] * unroll
    else: expected = ([mfma] + conv) * (unroll - 1) + conv[:split] + [mfma] + conv[split:]
    assert [op(l) for l in vec] == expected, key
    assert not any('load_' in l or 'store_' in l for l in body), key
    intervals = []; n = None
    for line in vec:
        if op(line) == mfma:
            if n is not None: intervals.append(n)
            n = 0
        else: n += 1
    intervals.append(n)
    assert intervals == ([0] * unroll if schedule == 0 else [8] * (unroll - 2) + [8 + split, 8 - split]), (key, intervals)
    # Track actual physical register phases through three cyclic iterations.
    # The first lap seeds the wraparound; subsequent laps must never consume
    # a partly converted register or overwrite the activation source.
    a_regs = set()
    for line in vec:
        if op(line) == mfma:
            lo, hi = sources(line)[0]; a_regs.update(range(lo, hi + 1))
    assert len(a_regs) == 2, key
    state = {r: 'a' if r in a_regs else 'ready' for r in range(256)}
    for lap in range(3):
        for line in vec:
            opcode = op(line)
            if opcode == mfma:
                a, b = sources(line)
                assert all(state[r] == 'a' for r in range(a[0], a[1] + 1)), (key, lap, line)
                assert all(state[r] == 'ready' for r in range(b[0], b[1] + 1)), (key, lap, line)
            else:
                dest = int(re.search(r'\bv(\d+),', line)[1])
                assert dest not in a_regs, (key, line)
                if opcode == 'v_perm_b32': state[dest] = 'permuted'
                elif opcode == 'v_pk_ashrrev_i16':
                    assert state[dest] == 'permuted', (key, line); state[dest] = 'shifted'
                elif opcode == 'v_and_b32':
                    assert state[dest] == 'shifted', (key, line); state[dest] = 'masked'
                elif opcode == 'v_pk_mul_f16':
                    assert state[dest] == 'masked', (key, line); state[dest] = 'ready'
                else: raise AssertionError((key, line))
    # Prelude >0 explicitly forces initialized AGPRs and a real load wait before
    # the timestamp. Independent VALUs / requested scalar NOP must stay there.
    before = [(pc, line) for pc, line in ins if pc < start]
    begin_time = next(pc for pc, line in ins if op(line) == 's_memtime')
    if prelude:
        init = [pc for pc, line in before if op(line) in ('v_accvgpr_write_b32', 'v_accvgpr_mov_b32')]
        assert len(init) == 16 and max(init) < begin_time, key
        pad_waits = [pc for pc, line in before if op(line) == 's_waitcnt' and 'vmcnt(0)' in line and max(init) < pc < begin_time]
        assert pad_waits, key
        xs = [line for pc, line in before if op(line) == 'v_xor_b32' and max(pad_waits) < pc < begin_time]
        assert len(xs) == {5: 2, 6: 4, 7: 8}.get(prelude, 0), (key, len(xs))
        if prelude in (2, 3, 4):
            imm = {2: 0, 3: 3, 4: 7}[prelude]
            assert any(max(pad_waits) < pc < begin_time and op(line) == 's_nop' and re.search(r's_nop\s+' + str(imm) + r'\b', line) for pc, line in before), key
    scalar = []; nm = nv = 0
    for line in body:
        opcode = op(line)
        if opcode == mfma: nm += 1
        elif opcode.startswith('v_'): nv += 1
        else:
            assert opcode in ('s_waitcnt', 's_nop', 's_add_i32', 's_cmp_ge_i32', 's_cmp_lt_i32', 's_cbranch_scc0', 's_cbranch_scc1'), (key, opcode)
            scalar.append({'instruction': line.strip().split('//')[0].strip(), 'preceding_mfmas': nm, 'preceding_valu': nv})
    header_ops = [s for s in scalar if s['instruction'].startswith(('s_waitcnt', 's_nop'))]
    if prelude:
        assert not header_ops, (key, header_ops)
    else:
        assert len(header_ops) == 1 and header_ops[0]['instruction'] == 's_waitcnt vmcnt(0)', (key, header_ops)
    block = next(b for b in re.split(r'(?=  - \.agpr_count:)', meta)
                 if re.search(r'\.name:\s*_Z12split_kernelILi' + str(schedule) + 'ELi' + str(prelude) + 'ELi' + str(unroll) + 'EE', b))
    fields = {name: int(re.search(r'\.' + name + r':\s*(\d+)', block)[1]) for name in
              ['agpr_count', 'vgpr_count', 'sgpr_count', 'private_segment_fixed_size', 'group_segment_fixed_size']}
    assert fields['private_segment_fixed_size'] == 0 and fields['group_segment_fixed_size'] == 0, key
    (folder / ('loop-' + '-'.join(map(str, key)) + '.txt')).write_text('\n'.join(body) + '\n')
    summary['kernels'].append({'schedule': schedule, 'prelude': prelude, 'unroll': unroll, 'resources': fields, 'interval_valu_counts': intervals, 'scalars': scalar, 'instructions': dict(Counter(op(l) for l in body))})
assert len(summary['kernels']) == 168, len(summary['kernels'])
(folder / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS all 168 entries: exact per-interval work, no copies/loop memory, cyclic source lifetimes, preheader init/wait/pad before clocks, no private/LDS bytes')
