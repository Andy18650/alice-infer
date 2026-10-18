#!/usr/bin/env python3
"""Audit the common kernel used by both grid shapes; paired queue safety."""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else 'temp/fp8-w8a16/grid')
isa=(folder/'grid-probe.isa').read_text(); meta=(folder/'metadata.txt').read_text()
mfma='v_mfma_f32_32x32x8f16'
def op(line): return re.sub(r'_e(?:32|64)$','',line.strip().split()[0])
def registers(text):
    result=set()
    for m in re.finditer(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]',text):
        if m[1] is not None: result.add(int(m[1]))
        else: result.update(range(int(m[2]),int(m[3])+1))
    return result
assert registers('v[18:19], v0, v[3:4]')=={0,3,4,18,19}
summary={'sha256':hashlib.sha256((folder/'grid-probe.hsaco').read_bytes()).hexdigest(),'kernels':[]}
rejected_early_read_tests=0
for section in re.split(r'(?=^0[0-9a-f]+ <)',isa,flags=re.M):
    match=re.search(r'<void grid_kernel<(\d+), (\d+), (\d+)>',section[:350])
    if not match: continue
    fmt,depth,unroll=map(int,match.groups()); key=fmt,depth,unroll
    streams=2
    ins=[]
    for line in section.splitlines():
        m=re.search(r'// ([0-9A-Fa-f]+):',line)
        if m: ins.append((int(m[1],16),line))
    candidates=[]
    for pc,line in ins:
        m=re.search(r'\bs_cbranch_\w+\s+(-?\d+)',line)
        if not m: continue
        off=int(m[1]); off=off-65536 if off>32767 else off
        start=pc+4+off*4
        body=[l for p,l in ins if start<=p<=pc]
        if start<pc and sum(mfma in l for l in body)==unroll: candidates.append((start,pc,body))
    assert candidates,key
    start,end,body=min(candidates,key=lambda x:len(x[2]))
    counts=Counter(op(l) for l in body)
    loads=[l for l in body if op(l).startswith('global_load_')]
    assert len(loads)==streams*unroll,key
    bload='global_load_dword' if fmt else 'global_load_dwordx2'
    expected_loads=['global_load_dwordx2',bload]*unroll
    assert [op(l) for l in loads]==expected_loads,key
    # Each inline pair must retain literal A-first/B-second issue ordering.
    for n,line in enumerate(body):
        if line in loads[::2]:
            assert op(body[n+1])==bload,(key,line)
    waits=[int(re.search(r'vmcnt\((\d+)\)',l)[1]) for l in body if 'vmcnt(' in l]
    assert waits==[streams*(depth-1)]*unroll,(key,waits)
    for opcode in ('v_perm_b32','v_pk_ashrrev_i16','v_pk_mul_f16'):
        assert counts[opcode]==(2*unroll if fmt else 0),(key,opcode)
    assert not any('store_' in op(l) or op(l).startswith('ds_') or op(l) in ('v_readlane_b32','v_writelane_b32') for l in body),key
    # Pessimistic FIFO: no request completes until a wait retires it. A read
    # before that wait is forbidden, including compiler-inserted phi copies.
    queue=deque(); peak=0
    def process(line):
        global peak
        opcode=op(line); asm=line.split('//')[0].strip()
        m=re.search(r's_waitcnt\s+.*vmcnt\((\d+)\)',asm)
        if m:
            while len(queue)>int(m[1]): queue.popleft()
            return
        pending=set().union(*queue) if queue else set()
        if opcode.startswith('v_') or opcode.startswith('global_load_'):
            args=asm[len(opcode):].strip() if asm.startswith(opcode) else asm.split(None,1)[1]
            first,_,rest=args.partition(',')
            if opcode==mfma or opcode.startswith('v_accvgpr_'):
                defs=set(); reads=registers(args)
            else:
                defs=registers(first); reads=registers(rest)
            assert not reads & pending,('premature read',key,line,reads & pending)
            assert not defs & pending,('premature overwrite',key,line,defs & pending)
            if opcode.startswith('global_load_'):
                queue.append(defs)
                peak=max(peak,len(queue))
    for pc,line in ins:
        if pc<start: process(line)
    initial=len(queue)
    assert initial==streams*(depth-1),(key,initial)
    if initial:
        # Reject a premature read from each stream, not just the first A load.
        for slot in range(streams):
            pending_reg=next(iter(queue[slot]))
            try: process(f'v_mov_b32 v255, v{pending_reg}')
            except AssertionError: rejected_early_read_tests+=1
            else: raise AssertionError(('early-read negative control failed',key,slot))
    # Reset peak after the prologue's independent activation/seed reads.
    peak=initial
    for lap in range(3):
        for line in body: process(line)
        assert len(queue)==initial,(key,lap,len(queue))
    assert peak==streams*depth,(key,peak)
    # The unused trailing requests must be drained before end timestamps.
    after=[(pc,l) for pc,l in ins if pc>end]
    end_time=next(pc for pc,l in after if op(l)=='s_memtime')
    assert any('s_waitcnt vmcnt(0)' in l for pc,l in after if pc<end_time),key
    suffix=''.join(f'Li{n}E' for n in key)
    block=next(b for b in re.split(r'(?=  - \.agpr_count:)',meta) if re.search(r'\.name:\s*_Z11grid_kernelI'+suffix+'E',b))
    fields={name:int(re.search(r'\.'+name+r':\s*(\d+)',block)[1]) for name in ('agpr_count','vgpr_count','sgpr_count','private_segment_fixed_size','group_segment_fixed_size')}
    assert fields['private_segment_fixed_size']==0 and fields['group_segment_fixed_size']==0,key
    (folder/('loop-'+'-'.join(map(str,key))+'.txt')).write_text('\n'.join(body)+'\n')
    summary['kernels'].append({'format':fmt,'depth':depth,'unroll':unroll,'initial_pending':initial,'peak_pending':peak,'resources':fields,'instructions':dict(counts)})
assert len(summary['kernels'])==16,len(summary['kernels'])
assert rejected_early_read_tests==24,rejected_early_read_tests
summary['premature_read_negative_controls_rejected']=rejected_early_read_tests
(folder/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS 16 shared kernel entries: A-first/B-second pairs, exact 2D-2 waits, three-lap register safety, 24 negative controls, endpoint drains, no private/LDS or lane spills')
