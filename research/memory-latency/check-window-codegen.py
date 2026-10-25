#!/usr/bin/env python3
"""Verify the actual independent burst size; do not infer it from C++ arrays."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('build');p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.build)
meta=next(root.glob('*.metadata.txt')).read_text();isa=next(root.glob('*.isa.txt')).read_text()
assert set(re.findall(r'\.private_segment_fixed_size:\s*(\d+)',meta))=={'0'}
assert set(re.findall(r'\.group_segment_fixed_size:\s*(\d+)',meta))=={'0'}
entries=[]
for section in re.split(r'(?=^[0-9a-f]+ <)',isa,flags=re.M):
 name=section.split('\n')[0]
 if 'window_entry<'not in name:continue
 w,d=map(int,re.search(r'window_entry<(\d+), (\d+)>',name).groups())
 lines=[x.split('//')[0].strip()for x in section.splitlines()[1:]if '// 'in x]
 sites=[i for i,x in enumerate(lines)if x.startswith('global_load_dword')]
 assert len(sites)==d
 for i in sites:assert ('dwordx4'in lines[i])==(w==4)and lines[i+1].startswith('v_add_u32')
 assert lines[sites[-1]+2]=='s_waitcnt vmcnt(0)'
 for x in lines[sites[0]:sites[-1]+2]:assert x.startswith(('global_load_dword','v_add_u32')),'hidden consume/wait inside burst'
 assert not any('s_memtime'in x or 's_memrealtime'in x for x in lines),'timing instrumentation'
 entries.append({'width_dwords':w,'depth':d,'load_sites':len(sites),'private_bytes':0,'lds_bytes':0,'entry':name})
assert len(entries)==10
Path(a.out).write_text(json.dumps({'entries':entries,'checks':'exact D independent loads with address-only adds, one full wait before consumption, no private/LDS or per-load timestamps'},indent=2)+'\n')
print('All ten burst kernels passed ISA/resource checks')
