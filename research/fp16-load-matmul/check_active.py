#!/usr/bin/env python3
"""Current direct-only regression matrix; historical copied paths are excluded."""
import argparse,csv,itertools,json,re,shlex,shutil,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];base=root/'temp/fp16-load-matmul'
out=Path(__file__).resolve().parent/'results/ready-cull'
for name in ('cases','builds','rejections'):(out/name).mkdir(parents=True,exist_ok=True)
configs=[(1,32,1),(2,32,1),(4,32,1),(8,32,1),(3,8,1),(16,1,1),(4,32,4)]
modes=('hot','l2','stream','lds-hot','lds-sweep');jobs=[];comparisons=[]
def build(d,u,s):return base/f'd{d}-u{u}-lds{s}-direct'
def norm(path):
    result=[]
    for line in path.read_text().splitlines():
        line=line.split('//')[0].strip()
        if line.startswith(('s_cbranch_','s_branch')):line=line.split()[0]+' TARGET'
        result.append(re.sub(r'\s+',' ',line))
    return result
for d,u,s in configs:
    folder=build(d,u,s);dest=out/'builds'/folder.name;dest.mkdir(exist_ok=True)
    for name in ('hashes.txt','compiler-flags.txt','inventory.json','operand-sequence.json'):shutil.copy2(folder/name,dest/name)
    old=base/f'd{d}-u{u}-lds{s}-verified-r0-waited'
    if not old.is_dir():old=base/f'd{d}-u{u}-lds{s}-verified-r0'
    for m,l in itertools.product(range(5),(1,2,4)):
        name=f'loop-m{m}-l{l}-w1.txt'
        comparisons.append(dict(build=folder.name,mode=m,loads=l,identical=norm(folder/name)==norm(old/name)))
    for mode,l,pattern in itertools.product(modes[3:] if s==4 else modes,(1,2,4),('legacy','hash')):
        jobs.append((d,u,s,mode,l,pattern,4096,0))
for d,u,n in ((3,8,40),(16,1,17)):
    for mode,l in itertools.product(modes,(1,2,4)):
        jobs.append((d,u,1,mode,l,'hash',n,7 if mode in ('l2','lds-sweep') else 0))
(out/'instruction-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
summary=[];rejections=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as manifest:
    for i,(d,u,s,mode,l,pattern,n,records) in enumerate(jobs):
        name=f'd{d}-u{u}-s{s}-{mode}-l{l}-{pattern}-n{n}'
        output=out/'cases'/(name+'.csv');log=out/'cases'/(name+'.log')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build(d,u,s)/'probe'),'--mode',mode,'--loads',str(l),
             '--verify','1','--pattern',pattern,'--steps',str(n),'--rounds','1','--warmup-ms','0']
        if records:cmd+=['--records',str(records)]
        manifest.write(shlex.join(cmd)+'\n');manifest.flush()
        if not(a.resume and log.exists() and 'COMPLETE verified=1' in log.read_text()):
            with output.open('w') as f,log.open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        row=list(csv.DictReader(output.open()))[-1]
        assert row['verified']==row['valid']==row['wait']=='1' and row['ready_banks']=='0'
        assert row['mismatch_count']==row['guard_errors']=='0'
        summary.append(dict(case=name,valid=1,max_abs_error=row['max_abs_error']))
        if i%25==0:print(f'{i+1}/{len(jobs)} exact cases',flush=True)
    def reject(cmd,code,text):
        proc=subprocess.run(cmd,capture_output=True,text=True)
        assert proc.returncode==code and text in proc.stderr and not proc.stdout,(cmd,proc)
        rejections.append(dict(command=shlex.join(cmd),exit=code,error=proc.stderr))
    for d,u,s in configs:reject([str(build(d,u,s)/'probe'),'--wait','0','--verify','0'],1,'wait-free execution was removed')
    here=Path(__file__).resolve().parent
    for ready in (0,1,2):
        reject(['python3',str(here/'generate.py'),'--ready-banks',str(ready),'--output',str(out/f'rejected-{ready}.hpp')],2,'unrecognized arguments')
        reject(['bash',str(here/'build.sh'),'2','32',str(out/'rejected-build'),'1',str(ready)],2,'Ready-bank selection was removed')
    for option in ('--module','--symbol'):
        reject([str(build(2,32,1)/'probe'),option,'legacy-replay'],1,'unknown option '+option)
subprocess.run(['python3',str(here/'run_matrix.py'),'--binary',str(build(2,32,1)/'probe'),
                '--output',str(out/'runner-smoke'),'--lds','--repeats','1','--rounds','1','--warmup-ms','0'],check=True)
smoke=list((out/'runner-smoke').glob('*-run1.csv'));assert len(smoke)==6
for file in smoke:
    row=list(csv.DictReader(file.open()))[-1];assert row['valid']==row['wait']=='1' and row['ready_banks']=='0'
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
(out/'rejections/summary.json').write_text(json.dumps(rejections,indent=2)+'\n')
text=(f'{len(summary)} exact full-output cases plus {len(smoke)} matrix-runner cases passed.\n'
      f'{len(rejections)} removed-option rejection checks passed.\n'
      f'{sum(x["identical"] for x in comparisons)}/{len(comparisons)} surviving instruction/register lists identical after address/branch normalization.\n'
      'Active kernel matrix: 5 address modes * 3 instruction shapes, direct raw consumption only; waits mandatory.\n')
(out/'summary.txt').write_text(text)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(text)
