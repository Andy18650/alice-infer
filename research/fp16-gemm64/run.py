#!/usr/bin/env python3
"""Exact verification first, then randomized whole-kernel timings."""
import argparse,csv,itertools,random,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];base=root/'temp/fp16-gemm64';out=Path(__file__).resolve().parent/'results/direct-only-persistent-regression'
for sub in ('verification','timing','builds'):(out/sub).mkdir(parents=True,exist_ok=True)
configs=[(2,32),(2,64),(2,128),(1,64),(3,64),(4,64)]
for d,t in configs:
    src=base/f'd{d}-u4-t{t}-direct-only';dst=out/'builds'/src.name;dst.mkdir(exist_ok=True)
    subprocess.run(['sha256sum','-c',str(src/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
    if a.resume and (dst/'hashes.txt').exists() and (dst/'hashes.txt').read_bytes()!=(src/'hashes.txt').read_bytes():
        raise RuntimeError('resume requires the original source/binary fingerprints: '+str(src))
    for name in ('hashes.txt','compiler-flags.txt','inventory.json'):shutil.copy2(src/name,dst/name)
rows=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    def run(kind,d,t,mode,buffers,k,sa,sb,steps=4096,records=0,repeat=0):
        producer='direct' if mode=='gemm' else 'none' # Metadata only.
        name=f'd{d}-t{t}-{mode}-{producer}-b{buffers}-k{k}-s{steps}-r{records}-a{sa}-b{sb}-run{repeat}'
        log=out/kind/(name+'.log');csvfile=out/kind/(name+'.csv')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(base/f'd{d}-u4-t{t}-direct-only'/'probe'),
             '--mode',mode,'--seed-a',str(sa),'--seed-b',str(sb),'--rounds','9' if kind=='timing' else '1',
             '--warmup-ms','200' if kind=='timing' else '0']
        if mode=='gemm':cmd+=['--buffers',str(buffers),'--k',str(k)]
        else:
            cmd+=['--steps',str(steps)]
            if records:cmd+=['--records',str(records)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        if not(a.resume and log.exists() and 'COMPLETE invalid=0 verified=1' in log.read_text()):
            with csvfile.open('w') as f,log.open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        samples=list(csv.DictReader(csvfile.open()))
        assert all(x['valid']=='1' and x['mismatches']==x['guards']=='0' for x in samples)
        if mode=='gemm':assert 'COPY CHECK' in log.read_text()
        # Final schema stores the absolute clocks for independent span audit.
        for x in samples:
            span=max(int(x[f'end{w}']) for w in range(4))-min(int(x[f'begin{w}']) for w in range(4))
            assert span==int(x['cta_cycles'])
        rows.append(dict(kind=kind,depth=d,stage_k=t,mode=mode,producer=producer,buffers=buffers,K=k,
                         seed_a=sa,seed_b=sb,steps=steps,records=records,repeat=repeat,samples=len(samples),
                         cycles_per_mfma=median(float(x['cycles_per_mfma']) for x in samples),
                         body_us=median(float(x['body_us']) for x in samples),
                         GFLOPs=median(float(x['logical_GFLOPs']) for x in samples)))
    for d,t in configs:
        keys=(0,1,2,7,8,9,31,32,33,63,64,65,127,128,129,255,257,1024,32768,65536) if (d,t)==(2,64) else \
             (1,7,t-1,t,t+1,2*t+1,1024,32768) if d==2 else (7,65,129,32768)
        seeds=((1729,2718),(1730,2719)) if d==2 else ((1729,2718),)
        for k,buffers,(sa,sb) in itertools.product(keys,(1,2),seeds):
            run('verification',d,t,'gemm',buffers,k,sa,sb)
        print('Verified GEMM build',d,t,flush=True)
    for buffers,(sa,sb) in itertools.product((1,2),((1730,2718),(1729,2719))):
        run('verification',2,64,'gemm',buffers,129,sa,sb)
    for d,(mode,steps,records),(sa,sb) in itertools.product((1,2,3,4),(('lds-hot',4096,1),('lds-sweep',4096,32),('lds-sweep',40,7)),((1729,2718),(1730,2719))):
        run('verification',d,64,mode,0,0,sa,sb,steps,records)
    for repeat in (1,2):
        jobs=list(itertools.product((32,64,128),(1,2)))
        random.Random(1729 if repeat==1 else 2718).shuffle(jobs)
        for t,buffers in jobs:
            run('timing',2,t,'gemm',buffers,32768,1729,2718,repeat=repeat)
            print('Timing',repeat,t,buffers,rows[-1]['cycles_per_mfma'],flush=True)
        for mode in ('lds-hot','lds-sweep'):run('timing',2,64,mode,0,0,1729,2718,records=1 if mode=='lds-hot' else 32,repeat=repeat)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
text=[f"Exact verified launches: {sum(x['samples'] for x in rows)}, all 4096 outputs and guards valid.",
      f"Verification-only cases: {sum(x['kind']=='verification' for x in rows)}; every GEMM case also checks all words in both LDS staging buffers.",
      'Primary metric: CTA core cycles per four-wave MFMA step, includes staging/compute/synchronization.',
      'stageK | buffers | cycles/MFMA run1/run2 | body us run1/run2 | GFLOPs run1/run2']
for t,buffers in itertools.product((32,64,128),(1,2)):
    r=[x for x in rows if x['kind']=='timing' and x['mode']=='gemm' and (x['stage_k'],x['buffers'])==(t,buffers)]
    text.append(f'{t} | {buffers} | '+' | '.join('/'.join(f'{x[f]:.3f}' for x in r) for f in ('cycles_per_mfma','body_us','GFLOPs')))
(out/'summary.txt').write_text('\n'.join(text)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(text))
