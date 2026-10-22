#!/usr/bin/env python3
"""Verify the64 intervals are load/VM0 timed, register-resident, unspilled."""
import json,re,sys
from pathlib import Path
out=Path(sys.argv[1]);isa=(out/'probe.isa').read_text();ops=[]
for line in isa.splitlines():
    m=re.search(r'^\s*(.*?)\s*//\s*[0-9A-Fa-f]+:',line)
    if m:ops.append(m[1])
loads=[i for i,s in enumerate(ops) if s.startswith('global_load_dword ') and s.endswith('lds')]
assert len(loads)==64,len(loads)
regs=[]
for i in loads:
    assert ops[i-2:i]==['s_memtime s[16:17]','s_waitcnt lgkmcnt(0)'],ops[i-2:i+6]
    assert ops[i+1:i+4]==['s_waitcnt vmcnt(0)','s_memtime s[18:19]','s_waitcnt lgkmcnt(0)'],ops[i:i+6]
    assert re.match(r's_sub_u32 s\d+, s18, s16$',ops[i+4]),ops[i+4]
    m=re.match(r'v_mov_b32_e32 v(\d+), s\d+$',ops[i+5]);assert m,ops[i+5];regs.append(int(m[1]))
assert len(set(regs))==64,'samples share a register'
body=ops[loads[0]-2:loads[-1]+6]
assert not any(re.search(r'global_store|flat_store|buffer_store|ds_read|ds_write|s_barrier',s) for s in body),body
assert sum('s_waitcnt vmcnt(0)' in s for s in body)==64
assert not re.search(r'scratch_|(?:read|write)lane',isa)
notes=(out/'metadata.txt').read_text();sizes=re.findall(r'\.private_segment_fixed_size:\s*(\d+)',notes);assert sizes and all(int(s)==0 for s in sizes)
(out/'timed-samples.txt').write_text('\n'.join(body)+'\n')
(out/'inventory.json').write_text(json.dumps(dict(samples_per_wave=64,vmem_loads=64,vm0_waits=64,
    register_destinations=regs,interleaved_result_stores=0,private_segment_bytes=0),indent=2)+'\n')
print('ISA:64 load/VM0 intervals,64 distinct VGPR captures; no result stores/LDS consumers/barriers inside, no scratch/lane spills.')
