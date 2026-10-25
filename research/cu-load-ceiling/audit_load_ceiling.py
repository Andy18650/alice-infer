#!/usr/bin/env python3
"""Audit pure-load rings: exact loads/waits per body, FIFO safety, no scratch."""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent.parent.parent/'temp/cu-load-ceiling')
isa=(folder/'load-ceiling-probe.isa').read_text(); meta=(folder/'metadata.txt').read_text()

def op(line): return re.sub(r'_e(?:32|64)$','',line.strip().split()[0])
def registers(text):
    result=set()
    for m in re.finditer(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]',text):
        if m[1] is not None: result.add(int(m[1]))
        else: result.update(range(int(m[2]),int(m[3])+1))
    return result
assert registers('v[18:19], v0, v[3:4]')=={0,3,4,18,19}

LOADS={0:{4:'global_load_dword',8:'global_load_dwordx2',16:'global_load_dwordx4'},
       3:{4:'ds_read_b32',8:'ds_read_b64',16:'ds_read_b128'}}
LOADS[4]=LOADS[0]   # hot2 uses the same global-load opcodes as hot
def expected_set(full):
    s=set()
    for w in (4,8,16): s.add((0,w,8,1,full))
    for b in (1,2,4,16): s.add((0,4,b,1,full))
    for w in (4,16):
        for d in (2,4,8,16,32,64): s.add((0,w,8,d,full))
    for w in ((4,16) if not full else (4,8,16)): s.add((4,w,8,1,full))
    for w in (4,8,16): s.add((3,w,8,1,full))
    return s
summary={'sha256':hashlib.sha256((folder/'load-ceiling-probe.hsaco').read_bytes()).hexdigest(),'kernels':[]}
rejected=0
found=set()
for section in re.split(r'(?=^0[0-9a-f]+ <)',isa,flags=re.M):
    match=re.search(r'<void load_kernel<(\d+), (\d+), (\d+), (\d+), (\d+)>',section[:400])
    if not match: continue
    mode,width,batch,dup,full=map(int,match.groups()); key=(mode,width,batch,dup,full)
    assert key not in found; found.add(key)
    lds=mode==3
    loader=LOADS[mode][width]
    ins=[]
    for line in section.splitlines():
        m=re.search(r'// ([0-9A-Fa-f]+):',line)
        if m: ins.append((int(m[1],16),line))
    # Timed body: backward branch containing exactly kUnroll(32) ring loads.
    candidates=[]
    for pc,line in ins:
        m=re.search(r'\bs_cbranch_\w+\s+(-?\d+)',line)
        if not m: continue
        off=int(m[1]); off=off-65536 if off>32767 else off
        start=pc+4+off*4
        body=[l for p,l in ins if start<=p<=pc]
        if start<pc and sum(op(l)==loader for l in body)==32: candidates.append((start,pc,body))
    assert candidates,key
    start,end,body=min(candidates,key=lambda x:len(x[2]))
    counts=Counter(op(l) for l in body)
    loads=[l for l in body if op(l)==loader]
    assert len(loads)==32,key
    counter='lgkmcnt' if lds else 'vmcnt'
    waits=[int(re.search(counter+r'\((\d+)\)',l)[1]) for l in body if counter+'(' in l]
    assert waits==[batch-1]*32,(key,waits[:4],len(waits))
    assert counts['v_xor_b32']==(32 if not full else 32*width//4),key
    forbidden={o for o in counts if o.startswith(('buffer_','global_store','ds_write')) or o in ('v_readlane_b32','v_writelane_b32') or 'mfma' in o}
    assert not forbidden,(key,forbidden)
    # Hot/LDS pointers must be loop-invariant VGPRs; mode 4 (hot2) alternates two.
    if mode in (0,3):
        def ptr(line): return registers(line.split('//')[0].split(',')[1])
        pointers=ptr(loads[0]); assert len(pointers)==(1 if lds else 2),key
        assert all(ptr(l)==pointers for l in loads),key
        for line in body:
            opcode=op(line)
            if opcode==loader or not opcode.startswith(('v_','global_load','ds_')): continue
            dest=registers(line.split('//')[0].split(None,1)[1].split(',')[0])
            assert not dest&pointers,('pointer rewritten',key,line)
    else:
        ptrs={frozenset(registers(l.split('//')[0].split(',')[1])) for l in loads}
        assert len(ptrs)<=2,(key,ptrs)
    # Pessimistic FIFO: nothing may read/overwrite a register whose load has
    # not been retired by a wait. Simulate prologue then three body laps.
    queue=deque(); peak=0
    def process(line):
        global rejected,peak
        opcode=op(line); asm=line.split('//')[0].strip()
        m=re.search(r's_waitcnt\s+.*'+counter+r'\((\d+)\)',asm)
        if m:
            while len(queue)>int(m[1]): queue.popleft()
            return
        pending=set().union(*queue) if queue else set()
        if opcode.startswith('v_') or opcode.startswith(('global_load','ds_')):
            asm_args=asm[len(opcode):].strip() if asm.startswith(opcode) else asm.split(None,1)[1]
            first,_,rest=asm_args.partition(',')
            defs=registers(first); reads=registers(rest)
            assert not reads&pending,('premature read',key,line)
            assert not defs&pending,('premature overwrite',key,line)
            if opcode==loader: queue.append(defs)
            peak=max(peak,len(queue))
    begin_time=next(pc for pc,l in ins if op(l)=='s_memtime')
    for pc,line in ins:
        if pc<start: process(line)
    initial=len(queue)
    assert initial==batch-1,(key,initial)
    for slot in range(initial):
        pending_reg=next(iter(queue[slot]))
        try: process(f'v_mov_b32 v255, v{pending_reg}')
        except AssertionError: rejected+=1
        else: raise AssertionError(('early-read negative control failed',key,slot))
    peak=initial
    for _ in range(3):
        for line in body: process(line)
        assert len(queue)==initial,(key,len(queue))
    assert peak==batch,(key,peak)
    after=[(pc,l) for pc,l in ins if pc>end]
    end_time=next(pc for pc,l in after if op(l)=='s_memtime')
    assert any(f's_waitcnt {counter}(0)' in l for pc,l in after if pc<end_time),key
    suffix='I'+''.join(f'Li{x}E' for x in key)
    block=next(b for b in re.split(r'(?=  - \.agpr_count:)',meta) if re.search(r'\.name:\s*_Z11load_kernel'+suffix,b))
    fields={name:int(re.search(r'\.'+name+r':\s*(\d+)',block)[1]) for name in ('vgpr_count','sgpr_count','private_segment_fixed_size','group_segment_fixed_size')}
    assert fields['private_segment_fixed_size']==0,key
    assert fields['group_segment_fixed_size']==(1024 if lds else 0),key
    (folder/('loop-'+'-'.join(map(str,key))+'.txt')).write_text('\n'.join(body)+'\n')
    summary['kernels'].append({'mode':mode,'width':width,'batch':batch,'dup':dup,'full':full,
                               'initial_pending':initial,'peak_pending':peak,'resources':fields,
                               'invariant_pointers':mode in (0,3)})
assert found==expected_set(0)|expected_set(1),found^expected_set(0)|expected_set(1)
assert rejected==sum(k['initial_pending'] for k in summary['kernels']),(rejected,)
section=next(s for s in re.split(r'(?=^0[0-9a-f]+ <)',isa,flags=re.M) if '<evict_cache(' in s[:300])
assert sum(op(l)=='global_load_dwordx4' for l in section.splitlines() if '//' in l)==1
assert 's_waitcnt vmcnt(0)' in section and 'v_mfma' not in section
for name in ('evict_cache','fill_values','fill_eviction'):
    block=next(b for b in re.split(r'(?=  - \.agpr_count:)',meta) if re.search(r'\.name:\s*\S*'+name,b))
    assert int(re.search(r'\.private_segment_fixed_size:\s*(\d+)',block)[1])==0
summary['premature_read_negative_controls_rejected']=rejected
summary['eviction_reader']='one explicit dwordx4 load per grid-stride iteration; runtime checksum validated'
(folder/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
# Keep reruns in the explicitly selected build/check directory; never rewrite
# the curated historical result set as a side effect of checking a new binary.
timed=[k for k in summary['kernels'] if not k['full']]
print(f'PASS {len(summary["kernels"])} kernels ({len(timed)} timed): exact 32-load bodies with per-ring wait discipline, '
      f'FIFO safety + {rejected} negative controls, invariant hot/LDS pointers, no scratch/LDS in global rings; eviction reader checked')
