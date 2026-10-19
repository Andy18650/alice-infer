#!/usr/bin/env python3
"""Regression checks after deleting wait-free generation and dispatch."""
import argparse,csv,itertools,json,re,shlex,shutil,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];base=root/'temp/fp16-load-matmul'
out=Path(__file__).resolve().parent/'results/waited-only'
for name in ('cases','builds','rejections'):(out/name).mkdir(parents=True,exist_ok=True)
configs=[(1,32,1,0),(2,32,1,0),(8,32,1,0),(3,8,1,0),(16,1,1,0),(4,32,4,0),(2,32,1,2)]
modes=('hot','l2','stream','lds-hot','lds-sweep');jobs=[];comparisons=[]
def build(d,u,s,r):return base/f'd{d}-u{u}-lds{s}-verified-r{r}-waited'
def normalize(path):
    result=[]
    for line in path.read_text().splitlines():
        line=line.split('//')[0].strip()
        if line.startswith(('s_cbranch_','s_branch')):line=line.split()[0]+' TARGET'
        result.append(re.sub(r'\s+',' ',line))
    return result
for d,u,s,r in configs:
    folder=build(d,u,s,r);dest=out/'builds'/folder.name;dest.mkdir(exist_ok=True)
    for name in ('hashes.txt','compiler-flags.txt','inventory.json','operand-sequence.json'):shutil.copy2(folder/name,dest/name)
    old=base/(f'd{d}-u{u}-lds{s}-verified-r{r}'+('-direct-control' if r else ''))
    for m,l in itertools.product(range(5),(1,2,4)):
        name=f'loop-m{m}-l{l}-w1.txt'
        comparisons.append(dict(build=folder.name,mode=m,loads=l,identical=normalize(folder/name)==normalize(old/name)))
    selected=modes[3:] if s==4 else modes
    for mode,l,pattern in itertools.product(selected,(1,2,4),('legacy','hash')):
        jobs.append((d,u,s,r,mode,l,pattern,4096,0))
for d,u,n in ((3,8,40),(16,1,17)):
    for mode,l in itertools.product(modes,(1,2,4)):
        jobs.append((d,u,1,0,mode,l,'hash',n,7 if mode in ('l2','lds-sweep') else 0))
(out/'instruction-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
summary=[];rejections=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    for i,(d,u,s,r,mode,l,pattern,n,records) in enumerate(jobs):
        name=f'd{d}-u{u}-s{s}-r{r}-{mode}-l{l}-{pattern}-n{n}'
        output=out/'cases'/(name+'.csv');log=out/'cases'/(name+'.log')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build(d,u,s,r)/'probe'),'--mode',mode,
             '--loads',str(l),'--verify','1','--pattern',pattern,'--steps',str(n),'--rounds','1','--warmup-ms','0']
        if records:cmd+=['--records',str(records)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        if not(a.resume and log.exists() and 'COMPLETE verified=1' in log.read_text()):
            with output.open('w') as f,log.open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        row=list(csv.DictReader(output.open()))[-1]
        assert row['verified']==row['valid']==row['wait']=='1'
        assert row['mismatch_count']==row['guard_errors']=='0'
        summary.append(dict(case=name,valid=1,max_abs_error=row['max_abs_error']))
        if i%25==0:print(f'{i+1}/{len(jobs)} exact cases',flush=True)
    for d,u,s,r in configs:
        path=build(d,u,s,r)/'probe'
        proc=subprocess.run([str(path),'--wait','0','--verify','0'],capture_output=True,text=True)
        assert proc.returncode==1 and 'wait-free execution was removed' in proc.stderr and not proc.stdout
        rejections.append(dict(binary=str(path),case='--wait 0 --verify 0',exit=proc.returncode,error=proc.stderr))
    old=base/'d2-u32-lds1'
    meta=(old/'metadata.txt').read_text()
    def symbol(wait):return re.search(r'\.name:\s*(_Z12matmul_probeILi0ELi1ELb'+str(wait)+r'EE\S+)',meta)[1]
    host=build(2,32,1,2)/'probe'
    cmd=[str(host),'--mode','hot','--loads','1','--module',str(old/'probe.hsaco'),'--symbol',symbol(0)]
    proc=subprocess.run(cmd,capture_output=True,text=True)
    assert proc.returncode==1 and 'wait-free symbols are disabled' in proc.stderr and not proc.stdout
    rejections.append(dict(binary=str(host),case='old wait-free module symbol',exit=proc.returncode,error=proc.stderr))
    safe=['numactl','--cpunodebind=0','--membind=0',str(host),'--mode','hot','--loads','1',
          '--module',str(old/'probe.hsaco'),'--symbol',symbol(1),'--wait','1','--rounds','1','--warmup-ms','0']
    commands.write(shlex.join(safe)+'\n')
    with (out/'cases/original-waited.csv').open('w') as f,(out/'cases/original-waited.log').open('w') as e:
        subprocess.run(safe,stdout=f,stderr=e,check=True)
    row=list(csv.DictReader((out/'cases/original-waited.csv').open()))[-1];assert row['valid']=='1'
    summary.append(dict(case='original-waited-module',valid=1,max_abs_error=row['max_abs_error']))
# Exercise current matrix enumeration (six LDS cases, no false variants).
subprocess.run(['python3',str(Path(__file__).resolve().parent/'run_matrix.py'),'--binary',str(build(2,32,1,0)/'probe'),
                '--output',str(out/'runner-smoke'),'--lds','--repeats','1','--rounds','1','--warmup-ms','0'],check=True)
smoke=list((out/'runner-smoke').glob('*-run1.csv'));assert len(smoke)==6
for f in smoke:
    row=list(csv.DictReader(f.open()))[-1];assert row['wait']==row['verified']==row['valid']=='1'
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
(out/'rejections/summary.json').write_text(json.dumps(rejections,indent=2)+'\n')
text=(f'{len(summary)} exact verified cases plus {len(smoke)} matrix-runner cases passed.\n'
      f'{len(rejections)} CLI/module wait-free rejection checks passed.\n'
      f'{sum(x["identical"] for x in comparisons)}/{len(comparisons)} surviving timed instruction/register lists identical after branch/address normalization.\n'
      '15 mandatory-wait entries per build; no generated wait-free variants. Historical raw samples preserved.\n')
(out/'summary.txt').write_text(text)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(text)
