#!/usr/bin/env python3
"""Copy-free raw-ring verification and fresh timing controls, all waited."""
import argparse,csv,itertools,random,shlex,shutil,subprocess
from pathlib import Path
from statistics import median

p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];base=root/'temp/fp16-load-matmul'
out=Path(__file__).resolve().parent/'results/direct-raw'
for name in ('verification','timing','builds'):(out/name).mkdir(parents=True,exist_ok=True)
summary=[];copied=set()
def folder(d,u,s,r):
    return base/(f'd{d}-u{u}-lds{s}-verified-r{r}'+('-direct-control' if r else ''))
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    def run(kind,d,u,s,r,mode,loads,pattern,sa,sb,steps,rounds,warmup,repeat=0,records=0,negative=False):
        build=folder(d,u,s,r)
        if build not in copied:
            dest=out/'builds'/build.name;dest.mkdir(exist_ok=True)
            for name in ('hashes.txt','compiler-flags.txt','inventory.json','operand-sequence.json'):shutil.copy2(build/name,dest/name)
            copied.add(build)
        stem=f'd{d}-u{u}-s{s}-r{r}-{mode}-l{loads}-{pattern}-a{sa}-b{sb}-n{steps}-run{repeat}'
        output=out/kind/(stem+'.csv');log=out/kind/(stem+'.log')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--mode',mode,'--loads',str(loads),
             '--wait','1','--verify','1','--pattern',pattern,'--seed-a',str(sa),'--seed-b',str(sb),
             '--steps',str(steps),'--rounds',str(rounds),'--warmup-ms',str(warmup)]
        if mode=='stream':records=steps+d+2 # same pitch/data for raw-only and R2.
        if records:cmd+=['--records',str(records)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        if not(a.resume and output.exists() and log.exists() and 'COMPLETE verified=1' in log.read_text()):
            with output.open('w') as f,log.open('w') as e:proc=subprocess.run(cmd,stdout=f,stderr=e,check=False)
            if proc.returncode not in (0,2):raise RuntimeError(f'execution error: {log}')
        rows=list(csv.DictReader(output.open()));invalid=sum(x['valid']!='1' for x in rows)
        summary.append(dict(kind=kind,depth=d,unroll=u,stride=s,ready_banks=r,mode=mode,loads=loads,
                            pattern=pattern,seed_a=sa,seed_b=sb,steps=steps,repeat=repeat,samples=len(rows),
                            expected_failure=int(negative),invalid=invalid,
                            mismatches=max(int(x['mismatch_count']) for x in rows),
                            guards=max(int(x['guard_errors']) for x in rows),max_error=max(float(x['max_abs_error']) for x in rows),
                            cycles_per_step=median(float(x['cta_cycles_per_step']) for x in rows)))
        if invalid and not negative:print('UNEXPECTED FAILURE',stem,summary[-1],flush=True)
    modes=('hot','l2','stream','lds-hot','lds-sweep')
    patterns=[('legacy',1729,2718),('hash',1729,2718),('hash',1730,2718),('hash',1729,2719),('hash',1730,2719)]
    configs=[(1,32,1),(2,32,1),(3,32,1),(4,32,1),(8,32,1),(4,32,4),(3,8,1),(16,1,1)]
    for d,u,s in configs:
        selected=modes if (d in (1,2,8) or u!=32) else modes[3:]
        for mode,loads,pattern in itertools.product(selected,(1,2,4),patterns):
            pat,sa,sb=pattern
            run('verification',d,u,s,0,mode,loads,pat,sa,sb,4096,1,0)
        print('Verified direct configuration',d,u,s,flush=True)
    for d in (1,2,8):
        for mode,loads in itertools.product(modes,(1,2,4)):
            run('verification',d,32,1,2,mode,loads,'hash',1729,2718,4096,1,0)
    for d,u,steps in ((3,8,40),(16,1,17)):
        for mode,loads in itertools.product(modes,(1,2,4)):
            run('verification',d,u,1,0,mode,loads,'hash',1730,2719,steps,1,0,records=7 if mode in ('l2','lds-sweep') else 0)
    for loads in (1,2,4):
        run('verification',2,32,1,1,'lds-sweep',loads,'hash',1729,2718,4096,1,0,negative=True)
    targets=[(d,1,0,mode,l) for d in (1,2,4) for mode in ('lds-hot','lds-sweep') for l in (1,2,4)]
    targets +=[(d,1,2,mode,l) for d in (1,2) for mode in ('lds-hot','lds-sweep') for l in (1,2,4)]
    targets +=[(8,1,r,'hot',l) for r in (0,2) for l in (1,2,4)]
    targets +=[(4,4,0,'lds-hot',l) for l in (1,2)]
    for repeat in (1,2):
        jobs=targets.copy();random.Random(1729 if repeat==1 else 2718).shuffle(jobs)
        for d,s,r,mode,loads in jobs:
            run('timing',d,32,s,r,mode,loads,'hash',1729,2718,4096,9,200,repeat)
            print('Timing',repeat,d,s,r,mode,loads,summary[-1]['cycles_per_step'],flush=True)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
positive=[x for x in summary if not x['expected_failure']]
negative=[x for x in summary if x['expected_failure']]
lines=[f"Checked positive launches: {sum(x['samples'] for x in positive)}, invalid: {sum(x['invalid'] for x in positive)}.",
       f"Copied-R1 negative controls with the same post-timestamp harvest guard: {sum(bool(x['invalid']) for x in negative)}/{len(negative)} fail.",
       'No copies/ready banks or extra arrival stage in R0. No padding between the load completion wait and MFMA.',
       'D | stride | R | mode | loads | cycles/step run1/run2']
for d,s,r,mode,l in targets:
    rows=[x for x in summary if x['kind']=='timing' and (x['depth'],x['stride'],x['ready_banks'],x['mode'],x['loads'])==(d,s,r,mode,l)]
    values='/'.join(f"{x['cycles_per_step']:.3f}" for x in rows)
    lines.append(f'{d} | {s} | {r} | {mode} | {l} | {values}')
(out/'summary.txt').write_text('\n'.join(lines)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(lines))
if any(x['invalid'] or x['guards'] for x in positive) or any(not x['invalid'] or x['guards'] for x in negative):raise SystemExit(1)
