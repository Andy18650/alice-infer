#!/usr/bin/env python3
"""Audit native64-load steady loop; optionally import loop extraction."""
import json,re,sys
from pathlib import Path
def instructions(isa):
    result=[]
    for line in isa.splitlines():
        m=re.search(r'^\s*(.*?)\s*//\s*([0-9a-fA-F]+):((?: [0-9a-fA-F]{8})+)',line)
        if m:result.append(dict(pc=int(m[2],16),asm=m[1],raw=b''.join(int(w,16).to_bytes(4,'little') for w in m[3].split())))
    return result
def loop_body(isa):
    ins=instructions(isa);found=[]
    for i,x in enumerate(ins):
        if not x['asm'].startswith('s_cbranch_scc1 '):continue
        d=int(x['asm'].split()[1],0);d=d-65536 if d>=32768 else d
        if d>=0:continue
        start=x['pc']+4+4*d;seq=[s for s in ins[:i+1] if s['pc']>=start]
        if sum(s['asm'].startswith('global_load_dword ') and s['asm'].endswith('lds') for s in seq)==64:found.append(seq)
    assert len(found)==1,len(found)
    return found[0]
def audit(out,wait):
    isa=(out/'probe.isa').read_text();seq=loop_body(isa);ops=[x['asm'] for x in seq]
    loads=[i for i,s in enumerate(ops) if s.startswith('global_load')]
    assert len(loads)==64 and not any(re.search(r's_barrier|ds_read|ds_write|s_nop',s) for s in ops)
    waits=[s for s in ops if s.startswith('s_waitcnt')]
    assert len(waits)==(0 if wait=='none' else 64)
    if wait!='none':
        assert all(ops[i+1].startswith(f's_waitcnt vmcnt({wait})') for i in loads)
    assert len(ops)==(81 if wait=='none' else 145),len(ops)
    assert sum(s.startswith('s_add_u32 m0') for s in ops)==4
    assert not re.search(r'scratch_|(?:read|write)lane',isa)
    sizes=re.findall(r'\.private_segment_fixed_size:\s*(\d+)',(out/'metadata.txt').read_text());assert sizes and all(int(s)==0 for s in sizes)
    (out/'loop.txt').write_text('\n'.join(ops)+'\n')
    (out/'inventory.json').write_text(json.dumps(dict(wait=wait,loads=64,loop_instructions=len(ops),per_load_waits=len(waits),scratch_bytes=0),indent=2)+'\n')
    print(f'ISA:64 direct LDS loads,{len(waits)} per-load waits({wait}),{len(ops)} loop instructions, no spills.')
if __name__=='__main__':audit(Path(sys.argv[1]),sys.argv[2])
