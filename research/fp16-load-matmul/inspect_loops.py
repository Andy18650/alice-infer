#!/usr/bin/env python3
"""Actual-code-object instruction and register inventories, not numerical tests."""
import argparse
from collections import Counter,deque
from pathlib import Path
import json,re

p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args(); folder=a.folder
header=(folder/'pipeline.hpp').read_text()
d,u,ph=map(int,re.search(r'kDepth=(\d+), kUnroll=(\d+), kPhases=(\d+)',header).groups())
stride=int(re.search(r'kLdsStride=(\d+)',header)[1])
isa=(folder/'probe.isa').read_text();metadata=(folder/'metadata.txt').read_text(); report=[]
def op(line): return re.sub(r'_e(?:32|64)$','',line.strip().split()[0])
def regs(text):
    result=set()
    for m in re.finditer(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]',text):
        result.update([int(m[1])] if m[1] else range(int(m[2]),int(m[3])+1))
    return result
for section in re.split(r'(?=^0[0-9a-f]+ <)',isa,flags=re.M):
    match=re.search(r'<void matmul_probe<(\d+), (\d+)>',section[:400])
    if not match:continue
    mode,loads=int(match[1]),int(match[2])
    ins=[l for l in section.splitlines() if re.search(r'// [0-9A-Fa-f]+:',l)]
    clocks=[i for i,l in enumerate(ins) if op(l)=='s_memtime']; assert len(clocks)==2
    # The copy-free loop starts with loads, not MFMA. Recover phase0 from
    # its backward conditional branch so no first-iteration loads are omitted.
    back=[]
    for i in range(clocks[0],clocks[1]):
        if (m:=re.search(r'\bs_cbranch_scc0\s+(-?\d+)',ins[i])):
            pc=int(re.search(r'// ([0-9a-fA-F]+):',ins[i])[1],16)
            off=int(m[1]);off=off-65536 if off>32767 else off
            if off<0:back.append(pc+4+4*off)
    assert len(back)==1
    start=next(i for i,l in enumerate(ins) if int(re.search(r'// ([0-9a-fA-F]+):',l)[1],16)==back[0])
    body=ins[start:clocks[1]]; counts=Counter(op(l) for l in body)
    lds=mode>=3; counter='lgkmcnt' if lds else 'vmcnt'
    n=ph*u;loader=({1:'ds_read_b128',2:'ds_read_b64',4:'ds_read_b32'} if lds else
                  {1:'global_load_dwordx4',2:'global_load_dwordx2',4:'global_load_dword'})[loads]
    assert counts['v_mfma_f32_32x32x8f16']==n
    assert sum(v for k,v in counts.items() if k.startswith(('global_load_','ds_read_')))==loads*n
    if lds: assert not any(op(l).startswith('global_load_') for l in ins)
    assert counts[loader]==loads*n
    assert counts['v_mov_b32']==0
    for i,line in enumerate(body):
        if op(line)=='v_mfma_f32_32x32x8f16':
            assert i>0 and op(body[i-1])=='s_waitcnt',('not direct wait->MFMA',mode,loads,line)
    mfmas=[l for l in body if op(l)=='v_mfma_f32_32x32x8f16']
    for step,line in enumerate(mfmas):
        args=line.split('//')[0].split(',')
        bank=4*(step%d)
        assert regs(args[1])==set(range(bank,bank+2))
        assert regs(args[2])==set(range(bank+2,bank+4))
    assert not counts['s_nop']
    # Harvest guard is after both ending counters, never in the measured loop.
    after=ins[clocks[1]:]
    wall_end=next(i for i,l in enumerate(after) if op(l)=='s_memrealtime')
    assert sum(op(l)=='s_nop' for l in after[wall_end+1:])==4
    assert counts['s_sub_u32']==ph and counts['s_cmp_eq_u32']==ph
    assert counts['s_cbranch_scc0']==ph
    assert counts['s_branch']==(ph if ph>1 else 0)
    threshold=min(loads*(d-1),14) if lds else loads*(d-1)
    waits=[int(re.search(counter+r'\((\d+)\)',l)[1]) for l in body if counter+'(' in l]
    assert waits==[threshold]*n+[0]
    assert counts['v_add_co_u32']==(n if mode in (1,2) else 0)
    assert counts['v_addc_co_u32']==(n if mode in (1,2) else 0)
    assert counts['v_add_u32']==(n if mode==4 else 0)
    suffix=f'ILi{mode}ELi{loads}EE'
    block=next(b for b in re.split(r'(?=  - \.agpr_count:)',metadata) if re.search(r'\.name:\s*_Z12matmul_probe'+suffix,b))
    resources={name:int(re.search(r'\.'+name+r':\s*(\d+)',block)[1]) for name in ('vgpr_count','agpr_count','sgpr_count','private_segment_fixed_size','group_segment_fixed_size')}
    assert resources['private_segment_fixed_size']==resources['group_segment_fixed_size']==0
    assert not any(op(l).startswith('buffer_') or (op(l).startswith('ds_') and op(l)!=loader)
                   or op(l) in ('v_readlane_b32','v_writelane_b32') for l in body)
    # All active entries wait. Conservatively retire pending registers FIFO.
    queue=deque()
    def process(line):
        opcode=op(line); text=line.split('//')[0].strip()
        if (m:=re.search(r'\b'+counter+r'\((\d+)\)',text)):
            while len(queue)>int(m[1]):queue.popleft()
        if opcode.startswith(('v_','global_load_','ds_read_')):
            args=text.split(None,1)[1];first,_,rest=args.partition(',')
            pending=set().union(*queue) if queue else set()
            assert not (regs(first)|regs(rest))&pending,('pending register read/overwrite',mode,loads,line)
            if opcode==loader or (not lds and opcode.startswith('global_load_')):queue.append(regs(first))
    for line in ins[:start]:process(line)
    assert len(queue)==loads*(d-1)
    for _ in range(2):
        for line in body[:-1]:process(line) # final instruction is the exit drain.
        assert len(queue)==threshold
    (folder/f'loop-m{mode}-l{loads}-w1.txt').write_text('\n'.join(body)+'\n')
    report.append(dict(mode=mode,loads=loads,wait=True,depth=d,unroll=u,phases=ph,lds_stride=stride,operand_storage='raw_ring',
                       counter=counter,effective_wait=threshold,priming_inside_timing=lds,
                       static_mfmas=n,counts=dict(counts),resources=resources))
assert len(report)==15
assert {(r['mode'],r['loads']) for r in report}=={(m,l) for m in range(5) for l in (1,2,4)}
(folder/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Inspected 15 mandatory-wait global/LDS entries: D={d}, U={u}, phases={ph}, direct raw ring, {ph*u} static MFMAs/entry, expected load/copy/wait/address counts, no scratch/static-LDS/lane spills. Dynamic LDS initialized outside timing, priming after timer waits. Waited FIFO lifetimes checked; no numerical tests.')
