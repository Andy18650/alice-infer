#!/usr/bin/env python3
"""Small-M screen with fixed parent work, no M partitioning.

K/fused: one CK producer grid covers all slices, then a deterministic combine.
N/separate: literal HIP stream controls; N also varies as a problem dimension.
Resume only completed jobs from the exact current executable generation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('out')
p.add_argument('--rounds',type=int,default=3)
p.add_argument('--repeats',type=int,default=16)
p.add_argument('--suite',choices=['k','n','both'],default='both')
p.add_argument('--cache',choices=['warm','evict'],default='warm')
p.add_argument('--m',type=int,nargs='+',default=[1,2,4,8,16,32,64])
p.add_argument('--n',type=int,nargs='+',default=[2048,8192,14336])
p.add_argument('--k',type=int,nargs='+',default=[4096,8192])
p.add_argument('--pads',type=int,nargs='+',default=[0,128])
p.add_argument('--q',type=int,nargs='+')
a=p.parse_args()
here=Path(__file__).resolve().parent
root=here.parent.parent
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256((root/'temp/gemm-granularity/bench').read_bytes()).hexdigest()
plan=[]
for axis in (['k','n'] if a.suite=='both' else [a.suite]):
    qs=a.q or ([1,2,4,8,16,32,64,128] if axis=='k' else [1,2,4,8])
    for q in qs:
        for pad in a.pads:
            for streams in ([1] if axis=='k' else sorted({1,min(q,4)})):
                name=f'{axis}-q{q}-s{streams}-pad{pad}-{a.cache}'
                dims=[]
                for m in a.m:
                    for n in a.n:
                        for k in a.k:
                            d=[m,n,k];i='mnk'.index(axis)
                            if d[i]%q:raise SystemExit('parent dimension must divide q exactly')
                            d[i]//=q;dims.append(d)
                shapes=out/f'{name}.shapes.csv'
                shapes.write_text(''.join(','.join(map(str,d))+'\n' for d in dims))
                cmd=['bash',str(here/'run.sh'),str(out/name),'small-m',
                     '--shapes',str(shapes),'--axis',axis,'--q',str(q),
                     '--streams',str(streams),'--b-pad',str(pad),
                     '--k-launch','fused' if axis=='k' else 'separate',
                     '--rounds',str(a.rounds),'--repeats',str(a.repeats),'--cache',a.cache]
                job={'name':name,'sha256':sha,'command':cmd,'shapes':dims}
                plan.append(job)
                marker=out/name/'complete.json'
                if marker.exists() and json.loads(marker.read_text())==job:
                    print('RESUME',name,flush=True);continue
                print('RUN',name,flush=True)
                subprocess.run(cmd,check=True)
                rows=list(csv.DictReader((out/name/'results.csv').open()))
                if not rows:raise SystemExit(f'empty run {name}')
                if 'ERROR:' in (out/name/'run.log').read_text():raise SystemExit(f'failed run {name}')
                # Completion markers are written only after child success; a networking
                # interruption cannot turn a partial CSV into an accepted completed job.
                marker.write_text(json.dumps(job,indent=2)+'\n')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with (out/'results.csv').open('w') as dest:
    writer=None
    for job in plan:
        with (out/job['name']/'results.csv').open() as file:
            reader=csv.DictReader(file)
            if writer is None:
                writer=csv.DictWriter(dest,fieldnames=reader.fieldnames,lineterminator='\n');writer.writeheader()
            writer.writerows(reader)
print(out/'results.csv',flush=True)
