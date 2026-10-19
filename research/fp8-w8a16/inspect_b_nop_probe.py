#!/usr/bin/env python3
"""Inspect MFMA-to-NOP ablation instruction sequences; no numerical tests."""
from collections import Counter
from pathlib import Path
import json
import re

root=Path('temp/fp8-w8a16')
loops={}; report=[]
for kind,foldername,filename in [('mfma','b-only','b-only-probe'),('nop','b-nop','b-nop-probe')]:
    for mode in ('waited','unwaited'):
        folder=root/foldername/mode
        text=(folder/f'{filename}.isa').read_text()
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
                target='v_mfma_f32_32x32x8f16' if kind=='mfma' else 's_nop'
                if start<pc and sum(l.strip().split()[0]==target for l in body)==unroll:
                    candidates.append(body)
            body=min(candidates,key=len)
            counts=Counter(l.strip().split()[0] for l in body)
            assert sum(v for k,v in counts.items() if k.startswith('global_load_'))==32
            assert counts['global_load_dword' if fmt else 'global_load_dwordx2']==32
            assert counts['s_waitcnt']==(32 if mode=='waited' else 0)
            for op in ('v_perm_b32','v_pk_ashrrev_i16','v_and_b32_e32','v_pk_mul_f16'):
                assert counts[op]==(64 if fmt else 0)
            assert counts['v_mov_b32_e32']==(0 if fmt else 64)
            if kind=='nop':
                assert counts['v_mfma_f32_32x32x8f16']==0
                assert all(re.search(r'\bs_nop\s+0\s*//',l) for l in body if l.strip().startswith('s_nop'))
                (folder/('loop-'+'-'.join(map(str,key))+'.txt')).write_text('\n'.join(body)+'\n')
            normalized=[]
            for line in body:
                line=line.split('//')[0].strip()
                if line.startswith(('v_mfma_f32_32x32x8f16','s_nop')): line='COMPUTE_SLOT'
                if line.startswith('s_cbranch_'): line=line.split()[0]+' LOOP'
                normalized.append(re.sub(r'\s+',' ',line))
            loops[kind,mode,key]=normalized
            report.append(dict(kind=kind,mode=mode,format=fmt,depth=depth,counts=dict(counts)))
keys={key for kind,mode,key in loops}
assert len(keys)==10
for mode in ('waited','unwaited'):
    for key in keys:
        before=loops['mfma',mode,key]; after=loops['nop',mode,key]
        if before!=after:
            diffs=[(i,a,b) for i,(a,b) in enumerate(zip(before,after)) if a!=b]
            raise AssertionError(('unintended instruction/register differences',mode,key,diffs[:8]))
(root/'b-nop/instruction-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print('20 NOP loops: 32 MFMAs replaced by 32 s_nop 0; all remaining instructions/registers identical to MFMA controls after branch normalization. B loads and preparation retained. No numerical tests.')
