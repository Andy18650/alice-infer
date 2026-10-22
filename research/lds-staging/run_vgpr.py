#!/usr/bin/env python3
"""One-CTA DWORDx4 VGPR ring,1/2/4 waves, two counterbalanced repeats."""
import argparse,csv,json,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/vgpr-ring');p.add_argument('--output',type=Path,default=here/'results/vgpr-ring');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve()
if out.exists() and any(out.iterdir()):p.error('choose a fresh results directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw','smoke'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('vgpr-ring.hip','generate-vgpr.py','build-vgpr.sh','inspect_vgpr.py','run_vgpr.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('vgpr-loads.hpp','compiler-flags.txt','hashes.txt','metadata.txt','loop.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),placement='numactl CPU+memory node0',requested_bytes_per_timing=1073741824,waves=[1,2,4],ring_slots=32,source_bytes=1048576),indent=2)+'\n')
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
rows=[]
with (out/'commands.txt').open('w') as commands:
    def run(waves,repeat,groups=None):
        smoke=groups is not None;folder='smoke' if smoke else 'raw';name=f'w{waves}-r{repeat}'+(f'-g{groups}' if smoke else '')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--waves',str(waves),'--rounds','1' if smoke else '9','--warmup-ms','0' if smoke else '200']
        if smoke:cmd+=['--groups',str(groups)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        with (out/folder/(name+'.csv')).open('w') as f,(out/folder/(name+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with (out/folder/(name+'.csv')).open() as f:data=list(csv.DictReader(f))
        assert len(data)==(1 if smoke else 9)
        assert 'every final ring DWORD/lane/slot and output guard exact' in (out/folder/(name+'.log')).read_text()
        for s in data:
            assert s['valid']=='1' and s['mismatches']==s['guards']=='0'
            assert int(s['waves'])==waves and int(s['threads'])==waves*64 and int(s['ring_slots'])==32 and int(s['source_bytes'])==1048576
            n=int(s['groups'])*32;assert int(s['loads_per_wave'])==n and int(s['requested_bytes'])==n*waves*1024
            if not smoke:assert int(s['requested_bytes'])==1073741824
            first=min(int(s[f'begin{w}']) for w in range(waves));end=max(int(s[f'issue{w}']) for w in range(waves));assert end-first==int(s['issue_cycles'])
            assert abs(float(s['cycles_per_wave_group_load'])-(end-first)/n)<1e-8
            assert all(int(s[f'{key}{w}'])==0 for w in range(waves,4) for key in ('begin','issue','done','wall_begin','wall_issue','wall_done'))
        if not smoke:
            row=dict(waves=waves,repeat=repeat,samples=9)
            for key in ('cycles_per_wave_group_load','requested_B_per_issue_cycle','requested_GBps','issue_us','completion_us','event_us','core_MHz'):row[key]=median(float(s[key]) for s in data)
            rows.append(row);print(row,flush=True)
    for waves in (1,2,4):
        for groups in (1,32//waves+1):run(waves,0,groups)
    for repeat,order in ((1,(1,2,4)),(2,(4,2,1))):
        for waves in order:run(waves,repeat)
with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('COMPLETE54 checked timing samples+6 fill/wrap smokes; all final ring DWORDs and guards exact.')
