#!/usr/bin/env python3
"""Check actual gfx90a load/wait/timestamp instructions and private metadata."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('build');p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.build);isa=next(root.glob('*.isa.txt')).read_text();metadata=next(root.glob('*.metadata.txt')).read_text()
assert set(re.findall(r'\.private_segment_fixed_size:\s*(\d+)',metadata))=={'0'},'private storage'
assert set(re.findall(r'\.group_segment_fixed_size:\s*(\d+)',metadata))=={'0'},'LDS storage'
entries=[]
for section in re.split(r'(?=^[0-9a-f]+ <)',isa,flags=re.M):
 title=section.split('\n')[0]
 if '<void chase<' not in title:continue
 instructions=[line.strip().split('//')[0].strip()for line in section.splitlines()[1:]if '// ' in line]
 control='true>(unsigned' in title
 loads=[i for i,line in enumerate(instructions)if line.startswith('global_load_')]
 assert len(loads)==(0 if control else 3),(title,loads)
 for i in loads:assert instructions[i+1]=='s_waitcnt vmcnt(0)',(title,i)
 if not control:
  i=loads[1]
  assert instructions[i-2].startswith('s_memtime ')and instructions[i-1]=='s_waitcnt lgkmcnt(0)'
  assert instructions[i+2].startswith('s_memtime ')and instructions[i+3]=='s_waitcnt lgkmcnt(0)'
  width4='chase<4,'in title
  assert all(('dwordx4' in instructions[i])==width4 for i in loads)
 entries.append({'entry':title,'control':control,'global_load_sites':len(loads),'core_timestamp_sites':sum(x.startswith('s_memtime ')for x in instructions),'steady_timestamp_sites':sum(x.startswith('s_memrealtime ')for x in instructions),'load_instructions':[instructions[i]for i in loads]})
assert len(entries)==7,len(entries)
result={'private_bytes':0,'lds_bytes':0,'entries':entries,'checks':'three live loads per read entry (warm/single/timed loop), immediate vmcnt(0), timed single bracket without intervening address arithmetic'}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print('ISA/resource checks passed:',len(entries),'entries')
