#!/usr/bin/env python3
"""Paired-format screen: fixed 32x128 storage tiles, no split-dependent packing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser()
p.add_argument('out');p.add_argument('--rounds',type=int,default=3)
p.add_argument('--suite',choices=['small','dense'],default='small')
p.add_argument('--q',nargs='+',type=int)
p.add_argument('--mode',choices=['graph','both'],default='graph')
a=p.parse_args();here=Path(__file__).resolve().parent;root=here.parent.parent
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256((root/'temp/gemm-granularity/bench').read_bytes()).hexdigest()
parents=([(m,n,k) for m in [1,4,16,64] for n in [2048,8192,14336] for k in [4096,8192]]
         if a.suite=='small' else [(512,4096,4096),(2048,8192,4096),(4096,8192,8192)])
qs=a.q or ([1,2,4,8,16,32] if a.suite=='small' else [1])
plan=[]
for q in qs:
    for fmt,pad in [('column',0),('column',128),('tile-n',0),('tile-k',0)]:
        name=f'{fmt}-p{pad}-q{q}'
        dims=[(m,n,k//q) for m,n,k in parents]
        if any(k%q for m,n,k in parents):raise SystemExit('parent K must divide q')
        shape_file=out/f'{name}.shapes.csv'
        shape_file.write_text(''.join(','.join(map(str,d))+'\n' for d in dims))
        cmd=['bash',str(here/'run.sh'),str(out/name),'tiled-'+a.suite,
             '--shapes',str(shape_file),'--axis','k','--q',str(q),'--k-launch','fused',
             '--weights',fmt,'--b-pad',str(pad),'--rounds',str(a.rounds),
             '--repeats','16','--mode',a.mode]
        job={'name':name,'sha256':sha,'command':cmd,'shapes':dims};plan.append(job)
        marker=out/name/'complete.json'
        if marker.exists() and json.loads(marker.read_text())==json.loads(json.dumps(job)):
            print('RESUME',name,flush=True);continue
        print('RUN',name,flush=True);subprocess.run(cmd,check=True)
        if 'ERROR:' in (out/name/'run.log').read_text():raise SystemExit('failed run '+name)
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
