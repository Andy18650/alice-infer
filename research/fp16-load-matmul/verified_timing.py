#!/usr/bin/env python3
"""Recheck key performance points, with exact output verification every pass."""
import csv,hashlib,json,re,shlex,shutil,subprocess
from pathlib import Path
from statistics import median

root=Path(__file__).resolve().parents[2];base=root/'temp/fp16-load-matmul'
out=Path(__file__).resolve().parent/'results/verified-timing';out.mkdir(parents=True,exist_ok=True)
jobs=[]
for loads in (1,2,4):
    jobs.extend([('original',8,1,'hot',loads,'legacy'),('original',1,1,'lds-hot',loads,'legacy'),
                 ('original',2,1,'lds-hot',loads,'legacy'),('original',2,1,'lds-sweep',loads,'legacy'),
                 ('original',4,4,'lds-hot',loads,'legacy'),('new',2,1,'lds-hot',loads,'hash')])
mode_ids={'hot':0,'lds-hot':3,'lds-sweep':4}
records=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as manifest:
    for run in (1,2):
        for backend,d,stride,mode,loads,pattern in (jobs if run==1 else reversed(jobs)):
            compiled=base/f'd{d}-u32-lds{stride}-verified'
            name=f'{backend}-d{d}-s{stride}-{mode}-l{loads}-{pattern}-run{run}'
            cmd=['numactl','--cpunodebind=0','--membind=0',str(compiled/'probe'),'--mode',mode,
                 '--loads',str(loads),'--wait','1','--verify','1','--pattern',pattern,'--rounds','9','--warmup-ms','200']
            module_hash=''
            if backend=='original':
                old=base/(f'd{d}-u32' if mode=='hot' else f'd{d}-u32-lds{stride}')
                symbol=re.search(r'\.name:\s*(_Z12matmul_probeILi'+str(mode_ids[mode])+r'ELi'+str(loads)+r'ELb1EE\S+)',(old/'metadata.txt').read_text())[1]
                module=old/'probe.hsaco';module_hash=hashlib.sha256(module.read_bytes()).hexdigest()
                cmd+=['--module',str(module),'--symbol',symbol]
            manifest.write(shlex.join(cmd)+'\n');manifest.flush()
            with (out/f'{name}.csv').open('w') as f,(out/f'{name}.log').open('w') as e:
                subprocess.run(cmd,stdout=f,stderr=e,check=True)
            rows=list(csv.DictReader((out/f'{name}.csv').open()))
            assert all(r['valid']=='1' and r['verified']=='1' and r['mismatch_count']==r['guard_errors']=='0' for r in rows)
            records.append(dict(backend=backend,depth=d,stride=stride,mode=mode,loads=loads,pattern=pattern,run=run,
                                samples=len(rows),cycles_per_step=median(float(r['cta_cycles_per_step']) for r in rows),module_sha256=module_hash))
            print(name,records[-1]['cycles_per_step'],flush=True)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(f'{sum(r["samples"] for r in records)} timed passes, every output exact and all guards intact.')
