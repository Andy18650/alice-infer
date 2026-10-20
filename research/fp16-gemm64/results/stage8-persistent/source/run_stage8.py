#!/usr/bin/env python3
"""Checked StageK8 / 2,4,8-stage pipeline sweep, explicit hash-bound binaries."""
import csv,itertools,json,random,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parents[2]
here=Path(__file__).resolve().parent
out=here/'results/stage8-persistent'
for sub in ('verification','timing','builds','source'):(out/sub).mkdir(parents=True,exist_ok=True)
build=root/'temp/fp16-gemm64/d2-u1-t8-direct-only'
control=root/'temp/fp16-gemm64/d2-u4-t128-direct-only'
for folder in (build,control):
    subprocess.run(['sha256sum','-c',str(folder/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
    dest=out/'builds'/folder.name;dest.mkdir(exist_ok=True)
    for name in ('hashes.txt','compiler-flags.txt','inventory.json','compute.hpp'):shutil.copy2(folder/name,dest/name)
for name in ('gemm64.hip','generate.py','data.hpp','build.sh','inspect_code.py','run_stage8.py'):shutil.copy2(here/name,out/'source'/name)
for folder in (build,control):
    dest=out/'builds'/folder.name
    for p in folder.glob('core-*.txt'):shutil.copy2(p,dest/p.name)
rows=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    def run(kind,buffers,k,sa=1729,sb=2718,repeat=0,mode='gemm',baseline=False):
        folder=control if baseline else build
        t=128 if baseline else 8
        name=f't{t}-b{buffers}-k{k}-a{sa}-b{sb}-run{repeat}-{mode}'
        cmd=['numactl','--cpunodebind=0','--membind=0',str(folder/'probe'),'--mode',mode,
             '--rounds','9' if kind=='timing' else '1','--warmup-ms','200' if kind=='timing' else '0',
             '--seed-a',str(sa),'--seed-b',str(sb)]
        if mode=='gemm':cmd+=['--buffers',str(buffers),'--k',str(k)]
        else:cmd+=['--steps','4096','--records','1' if mode=='lds-hot' else '32']
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        cf=out/kind/(name+'.csv');log=out/kind/(name+'.log')
        with cf.open('w') as f,log.open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with cf.open() as f:samples=list(csv.DictReader(f))
        assert len(samples)==(9 if kind=='timing' else 1)
        assert 'COMPLETE invalid=0 verified=1' in log.read_text()
        if mode=='gemm':assert 'COPY CHECK' in log.read_text()
        for x in samples:
            assert x['valid']=='1' and x['mismatches']==x['guards']=='0'
            assert int(x['stage_k'])==t and int(x['depth'])==2
            assert int(x['unroll'])==(4 if baseline else 1)
            assert int(x['outer_unroll'])==(buffers if mode=='gemm' and not baseline else 0)
            span=max(int(x[f'end{w}']) for w in range(4))-min(int(x[f'begin{w}']) for w in range(4))
            assert span==int(x['cta_cycles'])
            steps=int(x['steps']);assert abs(float(x['cycles_per_mfma'])-(span/steps if steps else 0))<1e-8
            if mode=='gemm':assert int(x['shared_bytes'])==buffers*t*256 and int(x['buffers'])==buffers
        rows.append(dict(kind=kind,stage_k=t,buffers=buffers,mode=mode,K=k,seed_a=sa,seed_b=sb,
                         repeat=repeat,samples=len(samples),
                         cycles_per_mfma=median(float(x['cycles_per_mfma']) for x in samples),
                         body_us=median(float(x['body_us']) for x in samples),
                         launch_us=median(float(x['launch_us']) for x in samples)))
    keys=(0,1,7,8,9,15,16,17,23,24,25,31,32,33,55,56,57,63,64,65,71,72,73,127,128,129,255,256,257,1024,32768,65535,65536)
    for stages in (2,4,8):
        for k,(sa,sb) in itertools.product(keys,((1729,2718),(1730,2718),(1729,2719))):run('verification',stages,k,sa,sb)
        print('Verified StageK8 stages',stages,flush=True)
    for repeat in (1,2):
        jobs=[(2,False),(4,False),(8,False),(2,True)]
        random.Random(1729+repeat).shuffle(jobs)
        for stages,baseline in jobs:
            run('timing',stages,32768,repeat=repeat,baseline=baseline)
            print('Timing',repeat,rows[-1],flush=True)
        for mode in ('lds-hot','lds-sweep'):run('timing',0,32768,repeat=repeat,mode=mode)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
text=[f"Exact checked launches: {sum(x['samples'] for x in rows)}; verification-only cases: {sum(x['kind']=='verification' for x in rows)}.",
      'Native O3 HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768, no model/context/clock changes.',
      'StageK8 raw D2 / outer unroll=stages; StageK128 control D2/U4. Two repeats, nine samples/process, 200 ms warmup.',
      'stageK | stages | cycles/four-wave MFMA step run1/run2 | total body us run1/run2 | full event us run1/run2']
for t,stages in ((8,2),(8,4),(8,8),(128,2)):
    r=[x for x in rows if x['kind']=='timing' and x['mode']=='gemm' and (x['stage_k'],x['buffers'])==(t,stages)]
    text.append(f'{t} | {stages} | '+' | '.join('/'.join(f'{x[f]:.3f}' for x in r) for f in ('cycles_per_mfma','body_us','launch_us')))
for mode in ('lds-hot','lds-sweep'):
    r=[x for x in rows if x['kind']=='timing' and x['mode']==mode]
    text.append(mode+' D1/U1 cycles/step: '+'/'.join(f"{x['cycles_per_mfma']:.3f}" for x in r))
(out/'summary.txt').write_text('\n'.join(text)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(text))
