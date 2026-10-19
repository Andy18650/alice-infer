#!/usr/bin/env python3
"""Instruction inventory only. No numerical or pipeline-safety validation."""
from collections import Counter
from pathlib import Path
import json
import re

root=Path('temp/fp8-w8a16/wait')
loops={}; report=[]
for mode in ('waited','unwaited'):
    folder=root/mode
    text=(folder/'wait-probe.isa').read_text()
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
        assert sum(v for k,v in counts.items() if k.startswith('global_load_'))==64
        assert counts['s_waitcnt']==(32 if mode=='waited' else 0)
        assert counts['v_pk_mul_f16']==(64 if fmt else 0)
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
print('20 streaming loops inspected: exact same instructions/registers after removing 32 waits and normalizing branch displacements. Loads, A moves and FP8 conversion retained. No math tests.')
