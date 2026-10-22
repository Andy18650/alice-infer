#!/usr/bin/env python3
"""Capture every raw interval and describe distributions without filtering."""
import argparse,csv,json,shlex,shutil,statistics,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/latency');p.add_argument('--output',type=Path,default=here/'results/latency');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve()
if out.exists() and any(out.iterdir()):p.error('choose a fresh output directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw','smoke'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('latency.hip','generate-latency.py','build-latency.sh','inspect_latency.py','run_latency.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('latency-records.hpp','compiler-flags.txt','hashes.txt','metadata.txt','timed-samples.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),batches=64,samples_per_wave_per_batch=64,placement='numactl CPU+memory node0',clock_overhead_subtracted=False),indent=2)+'\n')
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
def stats(values):
    q=statistics.quantiles(values,n=100,method='inclusive')
    return dict(n=len(values),min=min(values),max=max(values),median=statistics.median(values),mean=statistics.mean(values),
                stdev=statistics.pstdev(values),range=max(values)-min(values),p05=q[4],p25=q[24],p75=q[74],p95=q[94],p99=q[98],iqr=q[74]-q[24])
summary=[];waves=[];positions=[]
with (out/'commands.txt').open('w') as commands:
    def run(mode,repeat,batches=64,smoke=False):
        folder='smoke' if smoke else 'raw';name=f'{mode}-r{repeat}'
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--mode',mode,'--batches',str(batches),'--warmup-ms','0' if smoke else '200']
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        with (out/folder/(name+'.csv')).open('w') as f,(out/folder/(name+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with (out/folder/(name+'.csv')).open() as f:data=list(csv.DictReader(f))
        assert len(data)==batches*4*64
        assert {(int(x['batch']),int(x['wave']),int(x['sample'])) for x in data}=={(b,w,s) for b in range(batches) for w in range(4) for s in range(64)}
        assert all(int(x['source_offset'])==(int(x['batch'])%16)*16384 and x['mode']==mode for x in data)
        assert 'final full LDS/check guards and all sample guards pass' in (out/folder/(name+'.log')).read_text()
        if not smoke:
            summary.append(dict(mode=mode,repeat=repeat,**stats([int(x['cycles']) for x in data])))
            for w in range(4):waves.append(dict(mode=mode,repeat=repeat,wave=w,**stats([int(x['cycles']) for x in data if int(x['wave'])==w])))
            for s in range(64):positions.append(dict(mode=mode,repeat=repeat,sample=s,**stats([int(x['cycles']) for x in data if int(x['sample'])==s])))
    for mode in ('shared','private'):run(mode,0,batches=1,smoke=True)
    for repeat,modes in ((1,('shared','private')),(2,('private','shared'))):
        for mode in modes:run(mode,repeat)
for filename,rows in (('summary.csv',summary),('per-wave.csv',waves),('per-position.csv',positions)):
    with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('65536 raw timing-inclusive intervals +512 smoke intervals captured; no filtering/overhead subtraction. All LDS/sample guards pass.')
for row in summary:print(', '.join(f'{k}={v:.4f}' if isinstance(v,float) else f'{k}={v}' for k,v in row.items()))
