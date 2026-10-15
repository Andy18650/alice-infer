#!/usr/bin/env python3
"""Repeated dense GEMM storage tests with fixed-geometry and tuned comparisons.

No split-M/N/K: each operation is one GEMM. Screen a closed dense shortlist,
then confirm the plain-affine winner on all layouts and each layout's winner.
Selected cases are batched by configuration, avoiding per-case process startup.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

p=argparse.ArgumentParser()
p.add_argument('out');p.add_argument('--suite',choices=['screen','confirm'],default='screen')
p.add_argument('--screen');p.add_argument('--rounds',type=int,default=3)
p.add_argument('--cache',choices=['warm','evict'],default='warm')
p.add_argument('--seed',type=int,default=731)
a=p.parse_args();here=Path(__file__).resolve().parent;root=here.parent.parent
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
sha=hashlib.sha256((root/'temp/gemm-granularity/bench').read_bytes()).hexdigest()
layouts=[('column',0),('column',128),('tile-n',0),('tile-k',0)]
impls=['ck_64x64x64','ck_64x128x64','ck_128x128x64','ck_128x256x64','ck_256x256x64']
# TP4-sized 70B o / gup and representative 8B FFN dimensions.
parents=[(m,n,k) for m in [64,128,256,512,1024,2048,4096]
         for n,k in [(2048,8192),(8192,8192),(14336,8192),(8192,7168),(4096,14336),(28672,4096)]]
selected=collections.defaultdict(set);selection=[]
if a.suite=='screen':
    for fmt,pad in layouts:
        for impl in impls:selected[fmt,pad,impl].update(parents)
        if fmt=='column':selected[fmt,pad,'rocblas'].update(parents)
else:
    if not a.screen:raise SystemExit('--screen CSV is required for confirmation')
    groups=collections.defaultdict(list)
    for r in csv.DictReader(open(a.screen)):
        if r['mode']=='graph' and r['repeats']=='16':
            key=(tuple(int(r[k]) for k in ['m','n','k']),r['weights'],int(r['bpad']),r['impl'])
            groups[key].append(float(r['gpu_us']))
    timings={key:statistics.median(v) for key,v in groups.items()}
    for shape in parents:
        baseline=min((us,impl) for (s,fmt,pad,impl),us in timings.items()
                     if s==shape and fmt=='column' and pad==0 and impl.startswith('ck_'))[1]
        for fmt,pad in layouts:
            winner=min((us,impl) for (s,f,p,impl),us in timings.items()
                       if s==shape and f==fmt and p==pad and impl.startswith('ck_'))[1]
            selected[fmt,pad,baseline].add(shape);selected[fmt,pad,winner].add(shape)
            if fmt=='column':selected[fmt,pad,'rocblas'].add(shape)
            selection.append(dict(m=shape[0],n=shape[1],k=shape[2],weights=fmt,bpad=pad,
                                  fixed_impl=baseline,tuned_impl=winner))
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
jobs=list(selected.items());random.Random(a.seed).shuffle(jobs);plan=[]
for (fmt,pad,impl),shapes in jobs:
    dims=sorted(shapes);name=f'{fmt}-p{pad}-{impl}'
    shape_file=out/f'{name}.shapes.csv'
    shape_file.write_text(''.join(','.join(map(str,s))+'\n' for s in dims))
    cmd=['bash',str(here/'run.sh'),str(out/name),'prefill-'+a.suite,
         '--shapes',str(shape_file),'--weights',fmt,'--b-pad',str(pad),
         '--axis','k','--q','1','--k-launch','fused','--impl',impl,
         '--mode','both' if a.suite=='confirm' else 'graph','--cache',a.cache,
         '--rounds',str(a.rounds),'--repeats','16']
    job={'name':name,'sha256':sha,'command':cmd,'shapes':dims};plan.append(job)
    marker=out/name/'complete.json'
    if marker.exists() and json.loads(marker.read_text())==json.loads(json.dumps(job)):
        print('RESUME',name,flush=True);continue
    print('RUN',name,len(dims),'shapes',flush=True);subprocess.run(cmd,check=True)
    if 'ERROR:' in (out/name/'run.log').read_text():raise SystemExit('failed job '+name)
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
