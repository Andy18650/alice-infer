#!/usr/bin/env python3
"""Two counterbalanced process repetitions; nine verified samples each."""
import argparse,csv,json,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/base');p.add_argument('--output',type=Path,default=here/'results/base');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve()
if out.exists() and any(out.iterdir()):p.error('choose a fresh results directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw','smoke'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('probe.hip','generate.py','build.sh','inspect_code.py','run.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('loads.hpp','compiler-flags.txt','hashes.txt','metadata.txt','loop.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),output=str(out),groups=16384,placement='numactl CPU+memory node0'),indent=2)+'\n')
for when in ('before',):
    with (out/f'amd-smi-{when}.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
rows=[]
with (out/'commands.txt').open('w') as cmds:
    def run(mode,groups,repeat,smoke=False):
        folder='smoke' if smoke else 'raw';name=f'{mode}-g{groups}-r{repeat}'
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--mode',mode,'--groups',str(groups),
             '--rounds','1' if smoke else '9','--warmup-ms','0' if smoke else '200']
        cmds.write(shlex.join(cmd)+'\n');cmds.flush()
        with (out/folder/(name+'.csv')).open('w') as f,(out/folder/(name+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with (out/folder/(name+'.csv')).open() as f:data=list(csv.DictReader(f))
        assert len(data)==(1 if smoke else 9)
        for s in data:
            assert s['valid']=='1' and s['mismatches']==s['guards']=='0'
            assert int(s['source_bytes'])==1048576 and int(s['allocated_lds_bytes'])==65536
            begin=min(int(s[f'begin{w}']) for w in range(4));end=max(int(s[f'issue{w}']) for w in range(4));done=max(int(s[f'done{w}']) for w in range(4))
            assert end-begin==int(s['issue_cycles']) and done-begin==int(s['completion_cycles'])
            assert int(s['requested_bytes'])==groups*64*4*256
            assert abs(float(s['cycles_per_four_wave_load'])-(end-begin)/(groups*64))<1e-8
        if not smoke:
            row=dict(mode=mode,repeat=repeat)
            for key in ('cycles_per_four_wave_load','requested_B_per_issue_cycle','requested_GBps','issue_us','completion_us','event_us','core_MHz'):
                row[key]=median(float(s[key]) for s in data)
            rows.append(row)
    for mode in ('shared','private'):
        for groups in (1,16,17):run(mode,groups,0,True)
    for repeat,modes in ((1,('shared','private')),(2,('private','shared'))):
        for mode in modes:run(mode,16384,repeat)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('36 checked timing samples +6 smoke checks passed; source1MiB, allocated LDS64KiB, one CTA/256 threads, HIP0, CPU+memory NUMA0; no model/context/clock override.')
for mode in ('shared','private'):
    r=sorted([x for x in rows if x['mode']==mode],key=lambda x:x['repeat'])
    print(mode+': '+', '.join(key+'='+('/'.join(f'{x[key]:.4f}' for x in r)) for key in rows[0] if key not in ('mode','repeat')))
