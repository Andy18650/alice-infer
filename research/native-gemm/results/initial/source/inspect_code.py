#!/usr/bin/env python3
"""Check actual MFMA pipeline/resources, not just generated asm."""
import json,re,sys
from pathlib import Path
out=Path(sys.argv[1]);header=(out/'compute.hpp').read_text();T=int(re.search(r'kTile=(\d+)',header)[1]);R=T//64;P=R//2
isa=(out/'probe.isa').read_text();ops=[]
for line in isa.splitlines():
    m=re.search(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):',line)
    if m:ops.append((int(m[2],16),m[1]))
start=next(i for i,(_,s) in enumerate(ops) if s.startswith('s_memtime'))
end=next(i for i in range(start+1,len(ops)) if ops[i][1].startswith('s_memtime'))
body=[s for _,s in ops[start:end+1]]
assert sum(s.startswith('v_mfma_f32_32x32x8f16') for s in body)==16*R*R
assert sum(s.startswith('global_load_dwordx4') for s in body)==16*P
assert sum(s.startswith('ds_write_b128') for s in body)==12*P
assert sum(s.startswith('ds_read_b128') for s in body)==24*P
assert not any(re.search(r'accvgpr|scratch_|readlane|writelane|v_cvt',s) for s in body),body
notes=(out/'metadata.txt').read_text();assert re.search(r'\.private_segment_fixed_size:\s*0',notes),notes
assert re.search(r'\.group_segment_fixed_size:\s*0',notes)
assert not re.search(r'scratch_|(?:read|write)lane',isa)
loops=[]
for i,(pc,s) in enumerate(ops):
    if not s.startswith('s_branch '):continue
    d=int(s.split()[1],0);d=d-65536 if d>=32768 else d
    if d>=0:continue
    seq=[s for a,s in ops[:i+1] if a>=pc+4+4*d]
    if sum(s.startswith('v_mfma_') for s in seq)==8*R*R:loops.append(seq)
assert len(loops)==1,len(loops)
loop=loops[0];assert sum(s=='s_barrier' for s in loop)==4
assert sum(s.startswith('global_load_dwordx4') for s in loop)==8*P
assert sum(s.startswith('ds_read_b128') for s in loop)==16*P
assert sum(s.startswith('ds_write_b128') for s in loop)==8*P
assert not any('s_nop' in s for s in loop)
(out/'body.txt').write_text('\n'.join(body)+'\n');(out/'loop.txt').write_text('\n'.join(loop)+'\n')
fields={k:int(re.search(r'\.'+k+r':\s*(\d+)',notes)[1]) for k in ('vgpr_count','sgpr_count','private_segment_fixed_size')}
fields.update(tile=T,lds_bytes=T*128,MFMA_per_wave_per_two_K32_steps=8*R*R,barriers_per_two_steps=4,global_x4_per_two_steps=8*P,ds_write128_per_two_steps=8*P,ds_read128_per_two_steps=16*P)
(out/'inventory.json').write_text(json.dumps(fields,indent=2)+'\n')
print(fields)
