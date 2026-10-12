#!/usr/bin/env python3
"""Fixed small-M storage comparisons selected from the three-round screen."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--rounds',type=int,default=9)
p.add_argument('--cache',choices=['warm','evict'],default='warm');a=p.parse_args()
here=Path(__file__).resolve().parent;root=here.parent.parent
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256((root/'temp/gemm-granularity/bench').read_bytes()).hexdigest()
# m,n,parent-k,q,tile/family,format,padding. Every group includes combine.
cases=[]
for n,q in [(2048,8),(8192,2),(14336,4)]:
    for fmt,pad in [('column',0),('column',128),('tile-n',0),('tile-k',0)]:
        cases.append((1,n,8192,q,'ck_16x32x128',fmt,pad))
# Best screened CK and SIMT configurations, in addition to matched geometry.
cases += [(1,14336,8192,8,'ck_16x16x128','column',128),
          (1,8192,8192,1,'ck_16x16x128','column',128)]
for n,q in [(8192,1),(14336,4)]:
    for fmt,pad in [('column',128),('tile-k',0)]:
        cases.append((1,n,8192,q,'simt_gemv',fmt,pad))
for m,impl in [(16,'ck_16x32x128'),(64,'ck_64x64x64')]:
    for fmt,pad in [('column',128),('tile-k',0)]:cases.append((m,14336,8192,1,impl,fmt,pad))
plan=[]
for m,n,k,q,impl,fmt,pad in cases:
    name=f'm{m}-n{n}-k{k}-q{q}-{impl}-{fmt}-p{pad}'
    cmd=['bash',str(here/'run.sh'),str(out/name),'tiled-confirm',
         '--m',str(m),'--n',str(n),'--k',str(k//q),'--axis','k','--q',str(q),
         '--k-launch','fused','--weights',fmt,'--b-pad',str(pad),'--impl',impl,
         '--rounds',str(a.rounds),'--repeats','16','--cache',a.cache]
    job={'name':name,'sha256':sha,'command':cmd};plan.append(job);marker=out/name/'complete.json'
    if marker.exists() and json.loads(marker.read_text())==job:print('RESUME',name,flush=True);continue
    print('RUN',name,flush=True);subprocess.run(cmd,check=True)
    marker.write_text(json.dumps(job,indent=2)+'\n')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with (out/'results.csv').open('w') as dest:
    writer=None
    for job in plan:
        with (out/job['name']/'results.csv').open() as file:
            reader=csv.DictReader(file)
            if writer is None:writer=csv.DictWriter(dest,fieldnames=reader.fieldnames,lineterminator='\n');writer.writeheader()
            writer.writerows(reader)
print(out/'results.csv',flush=True)
