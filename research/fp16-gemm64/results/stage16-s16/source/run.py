#!/usr/bin/env python3
"""Focused StageK16 checks/timing, with an optional frozen StageK8 baseline."""
import argparse,csv,hashlib,itertools,json,re,shlex,shutil,subprocess,sys
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parents[2];here=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--build',type=Path,default=root/'temp/fp16-gemm64/stage16-s8')
p.add_argument('--output',type=Path,default=here/'results/stage16-s8')
p.add_argument('--full',action='store_true',help='144 winner-only verification cases instead of24 focused cases')
p.add_argument('--diagnostics',action='store_true',help='also check the independent LDS-only reference kernels')
p.add_argument('--baseline-build',type=Path,help='optional frozen StageK8 winner')
p.add_argument('--baseline-legacy-buffers',action='store_true',help='only for pre-cull binaries requiring --buffers16')
p.add_argument('--baseline-source',type=Path,help='frozen source directory for that archived binary')
a=p.parse_args();out=a.output.resolve();build=a.build.resolve()
old=a.baseline_build.resolve() if a.baseline_build else None
source=a.baseline_source.resolve() if a.baseline_source else None
if bool(old)!=bool(source):p.error('--baseline-build and --baseline-source must be supplied together')
if out.exists() and any(out.iterdir()):p.error('choose a fresh output directory; old measurements are preserved')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
if old:
    for line in (old/'hashes.txt').read_text().splitlines():
        sha,name=line.split(maxsplit=1);file=Path(name)
        if file.name in ('gemm64.hip','generate.py','data.hpp'):file=source/file.name
        assert hashlib.sha256(file.read_bytes()).hexdigest()==sha,('archived fingerprint',file)
for sub in ('verification','timing','builds','source'):(out/sub).mkdir(parents=True,exist_ok=True)
(out/'invocation.json').write_text(json.dumps(dict(argv=sys.argv,build=str(build),full=a.full,diagnostics=a.diagnostics),indent=2)+'\n')
for folder in [build]+([old] if old else []):
    dest=out/'builds'/folder.name;dest.mkdir()
    for name in ('hashes.txt','compiler-flags.txt','inventory.json','compute.hpp'):shutil.copy2(folder/name,dest/name)
    for file in folder.glob('core-*.txt'):shutil.copy2(file,dest/file.name)
for name in ('gemm64.hip','data.hpp','generate.py','inspect_code.py','build.sh','run.py'):shutil.copy2(here/name,out/'source'/name)
if source:shutil.copytree(source,out/'source-baseline')
def dimensions(folder):
    text=(folder/'compute.hpp').read_text()
    return tuple(int(re.search(r'\b'+key+r'=(\d+)',text)[1]) for key in ('kStageK','kStages'))
current_dims=dimensions(build);old_dims=dimensions(old) if old else None
rows=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    def run(kind,k,sa=1729,sb=2718,repeat=0,mode='gemm',steps=4096,records=1,baseline=False):
        label='archived-winner' if baseline else 'active'
        name=f'{label}-{mode}-k{k}-a{sa}-b{sb}-r{repeat}-s{steps}-w{records}'
        cmd=['numactl','--cpunodebind=0','--membind=0',str((old if baseline else build)/'probe'),
             '--mode',mode,'--rounds','9' if kind=='timing' else '1',
             '--warmup-ms','200' if kind=='timing' else '0','--seed-a',str(sa),'--seed-b',str(sb)]
        if mode=='gemm':
            cmd+=['--k',str(k)]
            if baseline and a.baseline_legacy_buffers:cmd+=['--buffers','16']
        else:cmd+=['--steps',str(steps),'--records',str(records)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        cf=out/kind/(name+'.csv');log=out/kind/(name+'.log')
        with cf.open('w') as f,log.open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with cf.open() as f:samples=list(csv.DictReader(f))
        assert len(samples)==(9 if kind=='timing' else 1)
        assert 'COMPLETE invalid=0 verified=1' in log.read_text()
        if mode=='gemm':assert 'COPY CHECK' in log.read_text()
        for x in samples:
            assert x['valid']=='1' and x['mismatches']==x['guards']=='0'
            stage,slots=old_dims if baseline else current_dims
            assert (int(x['depth']),int(x['unroll']),int(x['stage_k']))==(2,1,stage)
            span=max(int(x[f'end{w}']) for w in range(4))-min(int(x[f'begin{w}']) for w in range(4))
            assert span==int(x['cta_cycles'])
            n=int(x['steps']);assert abs(float(x['cycles_per_mfma'])-(span/n if n else 0))<1e-8
            if mode=='gemm':assert (int(x['buffers']),int(x['outer_unroll']),int(x['shared_bytes']))==(slots,slots*(stage//8),slots*stage*256)
        rows.append(dict(kind=kind,variant=label,stage_k=(old_dims if baseline else current_dims)[0],lds_slots=(old_dims if baseline else current_dims)[1],mode=mode,K=k,seed_a=sa,seed_b=sb,repeat=repeat,samples=len(samples),
                         cycles_per_mfma=median(float(x['cycles_per_mfma']) for x in samples),
                         body_us=median(float(x['body_us']) for x in samples)))
    if a.full:
        keys=(0,1,7,8,9,15,16,17,23,24,25,31,32,33,55,56,57,63,64,65,71,72,73,119,120,121,
              127,128,129,239,240,241,247,248,249,255,256,257,511,512,513,767,768,769,1024,32768,65535,65536)
        for k,(sa,sb) in itertools.product(keys,((1729,2718),(1730,2718),(1729,2719))):run('verification',k,sa,sb)
    else:
        keys=(0,1,8,9,15,16,17,31,32,33,127,128,129,255,256,257,511,512,513,32768,65535,65536)
        for k in keys:run('verification',k)
        for sa,sb in ((1730,2718),(1729,2719)):run('verification',257,sa,sb)
    if a.diagnostics:
        for mode,steps,records in (('lds-hot',4096,1),('lds-sweep',4096,32),('lds-sweep',40,7),('lds-sweep',5,3)):
            run('verification',0,mode=mode,steps=steps,records=records)
    for repeat in (1,2):
        jobs=[False,True] if old and repeat==1 else [True,False] if old else [False]
        for baseline in jobs:run('timing',32768,repeat=repeat,baseline=baseline)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
text=[f"Verification cases: {sum(x['kind']=='verification' for x in rows)}; timing samples: {sum(x['samples'] for x in rows if x['kind']=='timing')}; total checked samples: {sum(x['samples'] for x in rows)}.",
      'Native O3 HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768 timing, no model/context/clock changes.',
      f'Candidate: K{current_dims[0]}, raw2, {current_dims[1]} LDS slots, {current_dims[1]*current_dims[0]*256} LDS bytes; b64 reads unchanged.',
      'variant | cycles/four-wave MFMA step run1/run2 | entire body us run1/run2']
for label in ('active','archived-winner'):
    r=[x for x in rows if x['kind']=='timing' and x['variant']==label]
    if r:text.append(label+' | '+' | '.join('/'.join(f'{x[field]:.3f}' for x in r) for field in ('cycles_per_mfma','body_us')))
(out/'summary.txt').write_text('\n'.join(text)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(text))
