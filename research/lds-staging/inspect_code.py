#!/usr/bin/env python3
"""Audit actual native loop, not just inline-assembly source."""
import json,re,sys
from pathlib import Path
out=Path(sys.argv[1]);isa=(out/'probe.isa').read_text()
lines=[]
for line in isa.splitlines():
    m=re.search(r'^\s*(.*?)\s*//\s*([0-9A-Fa-f]+):',line)
    if m:lines.append((int(m[2],16),m[1]))
loops=[]
for i,(pc,op) in enumerate(lines):
    if op.startswith('s_cbranch_scc1'):
        # Recover target from signed SOPP displacement (in DWORDs).
        imm=int(op.split()[1],0);imm=imm-65536 if imm>=32768 else imm
        start=pc+4+4*imm
        seq=[s for addr,s in lines if start<=addr<=pc]
        if sum('global_load_dword ' in s and 'lds' in s for s in seq)==64:loops.append(seq)
assert len(loops)==1,len(loops)
loop=loops[0]
assert not any(re.search(r's_waitcnt|s_barrier|ds_read|ds_write|s_nop',s) for s in loop),loop
assert sum(s.startswith('s_add_u32 m0') for s in loop)==4
assert sum(s.startswith('v_add_co_u32') for s in loop)==4
assert sum(s.startswith('v_addc_co_u32') for s in loop)==4
assert sum(s.startswith('s_cbranch') for s in loop)==1
assert all('global_load_dword ' in s for s in loop if s.startswith('global_load'))
offsets=[int(re.search(r'offset:(\d+)',s)[1]) if 'offset:' in s else 0 for s in loop if s.startswith('global_load')]
assert offsets==list(range(0,4096,256))*4,offsets
assert not re.search(r'scratch_|(?:read|write)lane',isa)
notes=(out/'metadata.txt').read_text();assert re.search(r'\.private_segment_fixed_size:\s*0',notes)
# One explicit terminal VM0 after the backedge, not a loop throttle.
assert sum('s_waitcnt vmcnt(0)' in s for _,s in lines)==1
(out/'loop.txt').write_text('\n'.join(loop)+'\n')
(out/'inventory.json').write_text(json.dumps(dict(loads_per_wave_per_group=64,groups_4KiB=4,
    loop_instructions=len(loop),loop_waits=0,loop_barriers=0,loop_lds_consumers=0,
    source_bytes=1048576,allocated_lds_bytes=65536),indent=2)+'\n')
print(f'ISA:64 true LDS-target loads, {len(loop)} loop instructions, zero loop waits/barriers/consumers; terminal VM0 only.')
