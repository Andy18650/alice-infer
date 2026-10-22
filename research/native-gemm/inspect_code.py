#!/usr/bin/env python3
"""Audit actual native MFMA pipeline and no-spill resource contract."""
import collections,json,re,sys
from pathlib import Path
out=Path(sys.argv[1]);header=(out/'compute.hpp').read_text()
T=int(re.search(r'kTile=(\d+)',header)[1]);S=int(re.search(r'kStageK=(\d+)',header)[1]);R=T//64;P=R//2;Q=S//8
resident='kResident=true' in header;isa=(out/'probe.isa').read_text();ops=[]
for line in isa.splitlines():
    m=re.search(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):',line)
    if m:ops.append((int(m[2],16),m[1]))
start=next(i for i,(_,s) in enumerate(ops) if s.startswith('s_memtime'))
end=next(i for i in range(start+1,len(ops)) if ops[i][1].startswith('s_memtime'));body=[s for _,s in ops[start:end+1]]
def counts(seq):return collections.Counter(s.split()[0] for s in seq)
c=counts(body)
assert c['v_mfma_f32_32x32x8f16']==(1 if resident else 4)*Q*R*R
assert c['global_load_dwordx4']==(2 if resident else 4)*Q*P
assert c['ds_write_b128']==(1 if resident else 3)*Q*P
assert c['ds_read_b128']==(2 if resident else 6)*Q*P
assert not any(re.search(r'accvgpr|scratch_|readlane|writelane|v_cvt',s) for s in body)
notes=(out/'metadata.txt').read_text();assert re.search(r'\.private_segment_fixed_size:\s*0',notes),notes
assert re.search(r'\.group_segment_fixed_size:\s*0',notes)
assert not re.search(r'scratch_|(?:read|write)lane',isa)
loops=[]
for i,(pc,s) in enumerate(ops):
    if not s.startswith(('s_branch ','s_cbranch_scc1 ')):continue
    d=int(s.split()[1],0);d=d-65536 if d>=32768 else d
    if d>=0:continue
    seq=[s for a,s in ops[:i+1] if a>=pc+4+4*d]
    if counts(seq)['v_mfma_f32_32x32x8f16']==(1 if resident else 2)*Q*R*R:loops.append(seq)
assert len(loops)==1,len(loops)
loop=loops[0];c=counts(loop)
for op,want in (('s_barrier',4),('global_load_dwordx4',2*Q*P),('ds_read_b128',4*Q*P),('ds_write_b128',2*Q*P)):
    assert c[op]==(0 if resident else want),(op,c[op],want)
assert not any('s_nop' in s for s in loop)
(out/'body.txt').write_text('\n'.join(body)+'\n');(out/'loop.txt').write_text('\n'.join(loop)+'\n')
fields={k:int(re.search(r'\.'+k+r':\s*(\d+)',notes)[1]) for k in ('vgpr_count','sgpr_count','private_segment_fixed_size')}
fields.update(tile=T,stage_k=S,kind='resident-control' if resident else 'gemm',lds_bytes=T*S*4,instructions_per_loop=dict(c))
(out/'inventory.json').write_text(json.dumps(fields,indent=2)+'\n');print(fields)
