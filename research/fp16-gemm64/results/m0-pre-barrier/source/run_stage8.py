#!/usr/bin/env python3
"""Checked StageK8 ring sweep, explicit hash-bound binaries and fresh outputs."""
import argparse,csv,hashlib,itertools,json,random,shlex,shutil,subprocess,sys
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parents[2]
here=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--stages',type=int,nargs='+',default=[2,4,8])
p.add_argument('--verify-stages',type=int,nargs='+')
p.add_argument('--resume',action='store_true')
p.add_argument('--baseline-build',type=Path)
p.add_argument('--baseline-source',type=Path)
p.add_argument('--protocol',default='post-mfma-read')
p.add_argument('--baseline-protocol',default='retire-next')
p.add_argument('--build',type=Path,default=root/'temp/fp16-gemm64/d2-u1-t8-direct-only')
p.add_argument('--control',type=Path,default=root/'temp/fp16-gemm64/d2-u4-t128-direct-only')
p.add_argument('--output',type=Path,default=here/'results/stage8-persistent')
a=p.parse_args();out=a.output.resolve();build=a.build.resolve();control=a.control.resolve()
resume_tag=''
if a.resume:
    if not (out/'invocation.json').exists():p.error('resume requires an existing invocation')
    if json.loads((out/'invocation.json').read_text())['argv']!=[x for x in sys.argv if x!='--resume']:
        p.error('resume arguments must exactly match the original invocation')
    i=1
    while (out/f'invocation-resume{i}.json').exists():i+=1
    resume_tag=f'-resume{i}'
elif out.exists() and any(out.iterdir()):p.error('output is not empty; choose a fresh --output to preserve measurements')
for sub in ('verification','timing','builds','source'):(out/sub).mkdir(parents=True,exist_ok=True)
(out/f'invocation{resume_tag}.json').write_text(json.dumps(dict(argv=sys.argv,build=str(build),control=str(control),stages=a.stages),indent=2)+'\n')
timing_stages=sorted(set(a.stages+[8]+([16] if max(a.stages)>=32 else [])))
oldbuild=a.baseline_build.resolve() if a.baseline_build else None
oldsource=a.baseline_source.resolve() if a.baseline_source else None
if oldbuild and not oldsource:p.error('--baseline-build requires --baseline-source')
folders=[build,control]+([oldbuild] if oldbuild else [])
for folder in folders:
    if folder==oldbuild:
        for line in (folder/'hashes.txt').read_text().splitlines():
            digest,name=line.split(maxsplit=1);file=Path(name)
            if file.name in ('gemm64.hip','generate.py','data.hpp'):file=oldsource/file.name
            assert hashlib.sha256(file.read_bytes()).hexdigest()==digest,('baseline fingerprint',file)
    else:subprocess.run(['sha256sum','-c',str(folder/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
    dest=out/'builds'/folder.name;dest.mkdir(exist_ok=True)
    if a.resume:
        assert (dest/'hashes.txt').read_bytes()==(folder/'hashes.txt').read_bytes(),('changed resume fingerprint',folder)
    else:
        for name in ('hashes.txt','compiler-flags.txt','inventory.json','compute.hpp'):shutil.copy2(folder/name,dest/name)
if not a.resume:
    for name in ('gemm64.hip','generate.py','data.hpp','build.sh','inspect_code.py','run_stage8.py'):shutil.copy2(here/name,out/'source'/name)
    if oldsource:shutil.copytree(oldsource,out/'source-baseline')
    for folder in folders:
        dest=out/'builds'/folder.name
        for file in folder.glob('core-*.txt'):shutil.copy2(file,dest/file.name)
else:shutil.copy2(here/'run_stage8.py',out/'source'/f'run_stage8{resume_tag}.py')
rows=[]
with (out/f'amd-smi{resume_tag}-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/f'commands{resume_tag}.txt').open('w') as commands:
    def run(kind,buffers,k,sa=1729,sb=2718,repeat=0,mode='gemm',baseline=False,old=False):
        folder=oldbuild if old else control if baseline else build
        protocol='large-control' if baseline else a.baseline_protocol if old else a.protocol
        if mode!='gemm':protocol='lds-reference'
        t=128 if baseline else 8
        name=f'{protocol}-t{t}-b{buffers}-k{k}-a{sa}-b{sb}-run{repeat}-{mode}'
        cmd=['numactl','--cpunodebind=0','--membind=0',str(folder/'probe'),'--mode',mode,
             '--rounds','9' if kind=='timing' else '1','--warmup-ms','200' if kind=='timing' else '0',
             '--seed-a',str(sa),'--seed-b',str(sb)]
        if mode=='gemm':cmd+=['--buffers',str(buffers),'--k',str(k)]
        else:cmd+=['--steps','4096','--records','1' if mode=='lds-hot' else '32']
        cf=out/kind/(name+'.csv');log=out/kind/(name+'.log')
        reuse=a.resume and cf.exists() and log.exists() and 'COMPLETE invalid=0 verified=1' in log.read_text()
        commands.write(('# reused complete\n' if reuse else '# executed\n')+shlex.join(cmd)+'\n');commands.flush()
        if not reuse:
            if a.resume:
                for file in (cf,log):
                    if file.exists():file.rename(file.with_name(file.stem+resume_tag+'-interrupted'+file.suffix))
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
        rows.append(dict(kind=kind,protocol=protocol,stage_k=t,buffers=buffers,mode=mode,K=k,seed_a=sa,seed_b=sb,
                         repeat=repeat,samples=len(samples),
                         cycles_per_mfma=median(float(x['cycles_per_mfma']) for x in samples),
                         body_us=median(float(x['body_us']) for x in samples),
                         launch_us=median(float(x['launch_us']) for x in samples)))
    keys=(0,1,7,8,9,15,16,17,23,24,25,31,32,33,55,56,57,63,64,65,71,72,73,127,128,129,255,256,257,1024,32768,65535,65536)
    keys=tuple(sorted(set(keys+(119,120,121,239,240,241,247,248,249,511,512,513,767,768,769))))
    for stages in (a.verify_stages or a.stages):
        for k,(sa,sb) in itertools.product(keys,((1729,2718),(1730,2718),(1729,2719))):run('verification',stages,k,sa,sb)
        print('Verified StageK8 stages',stages,flush=True)
    for repeat in (1,2):
        jobs=[(s,False,False) for s in timing_stages]+[(2,True,False)]
        if oldbuild:jobs +=[(s,False,True) for s in timing_stages]
        random.Random(1729+repeat).shuffle(jobs)
        for stages,baseline,old in jobs:
            run('timing',stages,32768,repeat=repeat,baseline=baseline,old=old)
            print('Timing',repeat,rows[-1],flush=True)
        for mode in ('lds-hot','lds-sweep'):run('timing',0,32768,repeat=repeat,mode=mode)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
text=[f"Exact checked launches: {sum(x['samples'] for x in rows)}; verification-only cases: {sum(x['kind']=='verification' for x in rows)}.",
      'Native O3 HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768, no model/context/clock changes.',
      'StageK8 raw D2 / outer unroll=stages; StageK128 control D2/U4. Two repeats, nine samples/process, 200 ms warmup.',
      'protocol | stageK | stages | cycles/four-wave MFMA step run1/run2 | total body us run1/run2 | full event us run1/run2']
cases=[(a.protocol,8,s) for s in timing_stages]
if oldbuild:cases +=[(a.baseline_protocol,8,s) for s in timing_stages]
cases +=[('large-control',128,2)]
for protocol,t,stages in cases:
    r=[x for x in rows if x['kind']=='timing' and x['mode']=='gemm' and (x['protocol'],x['stage_k'],x['buffers'])==(protocol,t,stages)]
    text.append(f'{protocol} | {t} | {stages} | '+' | '.join('/'.join(f'{x[f]:.3f}' for x in r) for f in ('cycles_per_mfma','body_us','launch_us')))
for mode in ('lds-hot','lds-sweep'):
    r=[x for x in rows if x['kind']=='timing' and x['mode']==mode]
    text.append(mode+' D2/U1 cycles/step: '+'/'.join(f"{x['cycles_per_mfma']:.3f}" for x in r))
(out/'summary.txt').write_text('\n'.join(text)+'\n')
with (out/f'amd-smi{resume_tag}-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(text))
