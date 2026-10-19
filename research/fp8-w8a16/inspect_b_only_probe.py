#!/usr/bin/env python3
"""Instruction inventory only. No numerical or pipeline-safety validation."""
from collections import Counter
from pathlib import Path
import json
import re

root=Path('temp/fp8-w8a16/b-only')
loops={}; report=[]
for mode in ('waited','unwaited'):
    folder=root/mode
    text=(folder/'b-only-probe.isa').read_text()
    for section in re.split(r'(?=^0[0-9a-f]+ <)',text,flags=re.M):
        match=re.search(r'<void fixed_kernel<(\d+), (\d+), (\d+), (\d+)>',section[:350])
        if not match: continue
        key=tuple(map(int,match.groups())); fmt,fixed,depth,unroll=key
        ins=[(int(m[1],16),line) for line in section.splitlines() if (m:=re.search(r'// ([0-9a-fA-F]+):',line))]
        candidates=[]
        for pc,line in ins:
            if not (m:=re.search(r'\bs_cbranch_\w+\s+(-?\d+)',line)): continue
            off=int(m[1]); off=off-65536 if off>32767 else off
            start=pc+4+4*off
            body=[l for p,l in ins if start<=p<=pc]
            if start<pc and sum('v_mfma_f32_32x32x8f16' in l for l in body)==unroll:
                candidates.append(body)
        body=min(candidates,key=len)
        counts=Counter(l.strip().split()[0] for l in body)
        assert sum(v for k,v in counts.items() if k.startswith('global_load_'))==32
        assert counts['global_load_dword' if fmt else 'global_load_dwordx2']==32
        assert counts['s_waitcnt']==(32 if mode=='waited' else 0)
        waits=[int(re.search(r'vmcnt\((\d+)\)',l)[1]) for l in body if 'vmcnt(' in l]
        assert waits==([depth-1]*32 if mode=='waited' else [])
        for opcode in ('v_perm_b32','v_pk_ashrrev_i16','v_and_b32_e32','v_pk_mul_f16'):
            assert counts[opcode]==(64 if fmt else 0),(key,opcode)
        assert counts['v_mov_b32_e32']==(0 if fmt else 64)
        def regs(text):
            match=re.fullmatch(r'v\[(\d+):(\d+)\]',text.strip())
            return set(range(int(match[1]),int(match[2])+1)) if match else {int(text.strip()[1:])}
        mfm=[l for l in body if 'v_mfma_f32_32x32x8f16' in l]
        operands=[l.split('//')[0].split(',')[1].strip() for l in mfm]
        assert len(set(operands))==1,('A source changes',key)
        resident_a=regs(operands[0])
        loads=[l for l in body if l.strip().startswith('global_load_')]
        pointers=[l.split('//')[0].split(',')[1].strip() for l in loads]
        assert len(set(pointers))==1,('B address changes',key)
        protected=resident_a|regs(pointers[0])
        for line in body:
            opcode=line.strip().split()[0]
            if opcode.startswith(('v_','global_load_')) and not opcode.startswith('v_mfma_'):
                dest=line.strip().split(None,1)[1].split(',')[0]
                assert not regs(dest)&protected,('A or B pointer overwritten',key,line)
        (folder/('loop-'+'-'.join(map(str,key))+'.txt')).write_text('\n'.join(body)+'\n')
        # Strip encodings/addresses and branch displacement for exact comparison.
        normalized=[]
        for line in body:
            line=line.split('//')[0].strip()
            if line.startswith('s_waitcnt'): continue
            if line.startswith('s_cbranch_'): line=line.split()[0]+' LOOP'
            normalized.append(re.sub(r'\s+',' ',line))
        loops[mode,key]=normalized
        report.append(dict(mode=mode,format=fmt,depth=depth,counts=dict(counts)))
keys={key for mode,key in loops}
assert len(keys)==10
for key in keys:
    assert loops['waited',key]==loops['unwaited',key],('unexpected instruction/register change',key)
root.joinpath('instruction-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print('20 B-only loops inspected: 32 B loads, invariant resident A, no A writes/copies; FP8 conversion retained. Waited/unwaited loops identical except 32 waits/branch displacement. No math tests.')
