#!/usr/bin/env python3
"""Confirm fixed screening winners; never merge different executables/protocols."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
p=argparse.ArgumentParser()
p.add_argument('screen');p.add_argument('out')
p.add_argument('--rounds',type=int,default=9)
p.add_argument('--cache',choices=['warm','evict'],default='warm')
a=p.parse_args();screen=Path(a.screen);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
groups=collections.defaultdict(list)
for r in csv.DictReader(screen.open()):
    if r['mode']!='graph' or r['repeats']!='16' or not r['impl'].startswith('ck_'):continue
    m,n,k,q=map(int,(r['m'],r['n'],r['k'],r['q']))
    parent=(m,n*(q if r['axis']=='n' else 1),k*(q if r['axis']=='k' else 1))
    key=parent+(r['axis'],q,r['streams'],r['bpad'],r['impl'],r['k_launch'],r.get('weights','column'))
    groups[key].append(float(r['weight_GBps']))
winners={}
for key,values in groups.items():
    parent=key[:3];score=statistics.median(values)
    if parent not in winners or score>winners[parent][0]:winners[parent]=(score,key)
run=Path(__file__).with_name('run.sh').resolve();plan=[]
root=run.parent.parent.parent
sha=hashlib.sha256((root/'temp/gemm-granularity/bench').read_bytes()).hexdigest()
for parent,(_,key) in sorted(winners.items()):
    m,n,k=parent;axis,q,streams,pad,impl,launch,weights=key[3:]
    dims=list(parent);dims['mnk'.index(axis)]//=q
    name=f'{m}x{n}x{k}-{axis}-q{q}-s{streams}-p{pad}-{weights}'
    cmd=['bash',str(run),str(out/name),'small-m-confirm',
         '--m',str(dims[0]),'--n',str(dims[1]),'--k',str(dims[2]),
         '--axis',axis,'--q',str(q),'--streams',str(streams),'--b-pad',str(pad),
         '--k-launch',launch,'--impl',impl,'--weights',weights,'--cache',a.cache,
         '--rounds',str(a.rounds),'--repeats','16']
    job={'parent':parent,'configuration':key,'command':cmd,'sha256':sha};plan.append(job)
    marker=out/name/'confirmed.json'
    if marker.exists() and json.loads(marker.read_text())==json.loads(json.dumps(job)):
        print('RESUME',name,flush=True);continue
    print('RUN',name,impl,flush=True);subprocess.run(cmd,check=True)
    marker.write_text(json.dumps(job,indent=2)+'\n')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with (out/'results.csv').open('w') as dest:
    writer=None
    for job in plan:
        path=Path(job['command'][2])/'results.csv'
        with path.open() as file:
            reader=csv.DictReader(file)
            if writer is None:writer=csv.DictWriter(dest,fieldnames=reader.fieldnames,lineterminator='\n');writer.writeheader()
            writer.writerows(reader)
print(out/'results.csv',flush=True)
