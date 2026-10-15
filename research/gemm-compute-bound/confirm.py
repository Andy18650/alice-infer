#!/usr/bin/env python3
"""Repeat the already-working kernel; no new candidate search."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--binary',required=True)
p.add_argument('--rounds',type=int,default=9);p.add_argument('--pass-id',default='a')
a=p.parse_args();here=Path(__file__).resolve().parent;out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
binary=Path(a.binary).resolve();sha=hashlib.sha256(binary.read_bytes()).hexdigest()
cases=[(10,10,8192),(8,13,8192),(8,13,32768),(16,13,8192),(16,13,32768),
       (16,26,4096),(16,26,8192),(16,26,16384),(16,26,32768)]
if a.pass_id=='b':cases=list(reversed(cases))
plan=[]
for gm,gn,k in cases:
 name=f'{gm}x{gn}-k{k}'
 cmd=['bash',str(here/'run.sh'),str(out/name),str(binary),'--suite','explicit',
      '--impl','v3_256x256x64','--gm',str(gm),'--gn',str(gn),'--k',str(k),
      '--rounds',str(a.rounds),'--repeats','16']
 job={'sha256':sha,'command':cmd};plan.append(job);marker=out/name/'complete.json'
 if marker.exists() and json.loads(marker.read_text())==job:print('RESUME',name,flush=True);continue
 print('RUN',name,flush=True);subprocess.run(cmd,check=True);marker.write_text(json.dumps(job,indent=2)+'\n')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with (out/'results.csv').open('w') as dest:
 writer=None
 for job in plan:
  with (Path(job['command'][2])/'results.csv').open() as src:
   r=csv.DictReader(src)
   if writer is None:writer=csv.DictWriter(dest,fieldnames=r.fieldnames,lineterminator='\n');writer.writeheader()
   writer.writerows(r)
print(out/'results.csv',flush=True)
