#!/usr/bin/env python3
"""Check native ring register ownership and zero-wait steady loop."""
import json,re,sys
from pathlib import Path
out=Path(sys.argv[1]);isa=(out/'probe.isa').read_text();ins=[]
for line in isa.splitlines():
    m=re.search(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):',line)
    if m:ins.append((int(m[2],16),m[1]))
loops=[]
for i,(pc,op) in enumerate(ins):
    if not op.startswith('s_cbranch_scc1'):continue
    d=int(op.split()[1],0);d=d-65536 if d>=32768 else d
    if d>=0:continue
    seq=[s for addr,s in ins[:i+1] if addr>=pc+4+4*d]
    if sum(s.startswith('global_load_dwordx4 ') for s in seq)==32:loops.append(seq)
assert len(loops)==1,len(loops)
loop=loops[0];assert len(loop)==53,len(loop)
assert not any(re.search(r's_waitcnt|s_barrier|ds_|s_nop|v_mov|global_store|flat_store|buffer_store',s) for s in loop),loop
assert sum(s.startswith('v_add_co_u32') for s in loop)==8 and sum(s.startswith('v_addc_co_u32') for s in loop)==8
regs=[];slots=[];offsets=[]
for s in loop:
    if not s.startswith('global_load'):continue
    m=re.match(r'global_load_dwordx4 v\[(\d+):(\d+)\], v\[8:9\], off',s);assert m,s
    lo,hi=map(int,m.groups());assert hi-lo==3;slots.append((lo,hi));regs.extend(range(lo,hi+1))
    offsets.append(int(re.search(r'offset:(\d+)',s)[1]) if 'offset:' in s else 0)
assert len(set(regs))==128 and not {8,9}.intersection(regs)
assert offsets==[0,1024,2048,3072]*8
assert not re.search(r'scratch_|(?:read|write)lane',isa)
notes=(out/'metadata.txt').read_text();assert re.search(r'\.private_segment_fixed_size:\s*0',notes)
assert re.search(r'\.group_segment_fixed_size:\s*0',notes)
assert sum('s_waitcnt vmcnt(0)' in s for _,s in ins)==1
(out/'loop.txt').write_text('\n'.join(loop)+'\n')
(out/'inventory.json').write_text(json.dumps(dict(ring_slots=32,payload_vgprs=128,slots=slots,loop_instructions=53,loads_per_group=32,periodic_waits=0,lds_bytes=0,scratch_bytes=0),indent=2)+'\n')
print('ISA:32 disjoint DWORDx4 VGPR slots/128 payload registers,53 loop instructions, no timed waits/consumers/copies, no LDS or spills.')
