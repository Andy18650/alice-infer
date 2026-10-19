#!/usr/bin/env python3
"""One-ready-bank diagnostic: keep copies, verify math, compare fresh controls."""
import argparse,csv,itertools,random,shlex,shutil,subprocess
from pathlib import Path
from statistics import median

p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2]
base=root/'temp/fp16-load-matmul';out=Path(__file__).resolve().parent/'results/single-ready'
for sub in ('verification','timing','builds'): (out/sub).mkdir(parents=True,exist_ok=True)
configs=[(d,32,r) for d in (1,2,8) for r in (1,2)]+[(3,8,1)]
for d,u,r in configs:
    folder=base/f'd{d}-u{u}-lds1-verified-r{r}'
    dest=out/'builds'/folder.name;dest.mkdir(exist_ok=True)
    for name in ('hashes.txt','compiler-flags.txt','inventory.json'):shutil.copy2(folder/name,dest/name)
summary=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    def run(kind,d,u,r,mode,loads,pattern,sa,sb,steps,rounds,warmup,repeat=0,records=0):
        stem=f'd{d}-u{u}-r{r}-{mode}-l{loads}-{pattern}-a{sa}-b{sb}-n{steps}-run{repeat}'
        output=out/kind/(stem+'.csv');log=out/kind/(stem+'.log')
        folder=base/f'd{d}-u{u}-lds1-verified-r{r}'
        cmd=['numactl','--cpunodebind=0','--membind=0',str(folder/'probe'),'--mode',mode,
             '--loads',str(loads),'--wait','1','--verify','1','--pattern',pattern,'--seed-a',str(sa),
             '--seed-b',str(sb),'--steps',str(steps),'--rounds',str(rounds),'--warmup-ms',str(warmup)]
        # Match stream pitch/data across R1 and R2, including padding.
        if mode=='stream': records=steps+d+2
        if records:cmd+=['--records',str(records)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        if not(a.resume and output.exists() and log.exists() and 'COMPLETE verified=1' in log.read_text()):
            with output.open('w') as f,log.open('w') as e:
                proc=subprocess.run(cmd,stdout=f,stderr=e,check=False)
            if proc.returncode not in (0,2):raise RuntimeError(f'execution error: {log}')
        rows=list(csv.DictReader(output.open()))
        invalid=sum(row['valid']!='1' for row in rows)
        summary.append(dict(kind=kind,depth=d,unroll=u,ready_banks=r,mode=mode,loads=loads,pattern=pattern,
                            seed_a=sa,seed_b=sb,steps=steps,repeat=repeat,samples=len(rows),invalid=invalid,
                            mismatch_count=max(int(x['mismatch_count']) for x in rows),
                            guard_errors=max(int(x['guard_errors']) for x in rows),
                            max_abs_error=max(float(x['max_abs_error']) for x in rows),
                            cycles_per_step=median(float(x['cta_cycles_per_step']) for x in rows)))
        if invalid:print('INVALID',stem,summary[-1],flush=True)
    modes=('hot','l2','stream','lds-hot','lds-sweep')
    for d in (1,2,8):
        for r,mode,loads,pat in itertools.product((1,2),modes,(1,2,4),('legacy','hash','hash2')):
            seed=pat=='hash2'
            run('verification',d,32,r,mode,loads,'hash' if pat.startswith('hash') else 'legacy',1729+seed,2718+seed,4096,1,0)
        print('Verification depth',d,'complete',flush=True)
    for mode,loads in itertools.product(modes,(1,2,4)):
        run('verification',3,8,1,mode,loads,'hash',1730,2719,40,1,0,records=7 if mode in ('l2','lds-sweep') else 0)
    # Latency-exposed D1 plus deeply pipelined LDS/global controls, no unsafe reads.
    targets=[(d,mode,l) for d in (1,2) for mode in ('lds-hot','lds-sweep') for l in (1,2)]
    targets += [(8,'hot',l) for l in (1,2,4)]
    for repeat in (1,2):
        jobs=list(itertools.product(targets,(1,2)))
        random.Random(1729 if repeat==1 else 2718).shuffle(jobs)
        for (d,mode,loads),r in jobs:
            run('timing',d,32,r,mode,loads,'hash',1729,2718,4096,9,200,repeat)
            print('Timing',repeat,d,mode,loads,'R',r,summary[-1]['cycles_per_step'],flush=True)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
lines=[f"Exact output/guard checks: {sum(x['samples'] for x in summary)} checked launches; invalid {sum(x['invalid'] for x in summary)}.",
       'Copies retained: this tests immediate v_mov -> next MFMA, not copy removal.',
       'D | mode | loads | R1 run1/run2 | R2 run1/run2']
for d,mode,loads in targets:
    cells=[]
    for r in (1,2):
        vals=[x['cycles_per_step'] for x in summary if x['kind']=='timing' and (x['depth'],x['mode'],x['loads'],x['ready_banks'])==(d,mode,loads,r)]
        failed=sum(x['invalid'] for x in summary if x['kind']=='timing' and (x['depth'],x['mode'],x['loads'],x['ready_banks'])==(d,mode,loads,r))
        cells.append(('INVALID ' if failed else '')+'/'.join(f'{x:.3f}' for x in vals))
    lines.append(f'{d} | {mode} | {loads} | '+' | '.join(cells))
(out/'summary.txt').write_text('\n'.join(lines)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(lines))
if any(x['invalid'] for x in summary):raise SystemExit(1)
