#!/usr/bin/env python3
"""Coarse knee survey, three samples/repetition; preserve raw clocks/checks."""
import argparse,csv,json,re,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/grid-vgpr');p.add_argument('--output',type=Path,default=here/'results/grid-vgpr');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve();counts=(1,2,4,8,16,32,64,104,208)
if out.exists() and any(out.iterdir()):p.error('choose a fresh results directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw','smoke'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('grid-vgpr.hip','generate-vgpr.py','build-grid-vgpr.sh','inspect_vgpr.py','run_grid_vgpr.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('vgpr-loads.hpp','compiler-flags.txt','hashes.txt','metadata.txt','loop.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),ctas=counts,waves=[1,2,4],rounds=3,repetitions=2,requested_bytes_per_cta=268435456,source_bytes_per_cta=1048576,placement='numactl CPU+memory node0'),indent=2)+'\n')
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
rows=[]
with (out/'commands.txt').open('w') as commands:
    def run(ctas,waves,repeat,smoke=False):
        stem=f'n{ctas}-w{waves}-r{repeat}';folder='smoke' if smoke else 'raw'
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--ctas',str(ctas),'--waves',str(waves),'--rounds','1' if smoke else '3','--warmup-ms','0' if smoke else '100']
        if smoke:cmd+=['--groups','1']
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        with (out/folder/(stem+'.csv')).open('w') as f,(out/folder/(stem+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with (out/folder/(stem+'.csv')).open() as f:data=list(csv.DictReader(f))
        assert len(data)==ctas*(1 if smoke else 3)
        log=(out/folder/(stem+'.log')).read_text();assert 'every final ring DWORD/lane/slot and output guard exact' in log
        periods=[];bandwidth=[];done_bandwidth=[];event_bw=[];overlaps=[]
        for r in range(1 if smoke else 3):
            v=[x for x in data if int(x['round'])==r];assert {int(x['cta']) for x in v}==set(range(ctas))
            starts=[];ends=[];dones=[]
            for x in v:
                assert x['valid']=='1' and x['mismatches']==x['guards']=='0'
                assert int(x['ctas'])==ctas and int(x['waves'])==waves and int(x['ring_slots'])==32
                n=int(x['groups'])*32;assert int(x['loads_per_wave'])==n and int(x['requested_bytes'])==n*waves*1024
                if not smoke:assert int(x['requested_bytes'])==268435456
                b=min(int(x[f'begin{w}']) for w in range(waves));e=max(int(x[f'issue{w}']) for w in range(waves));assert int(x['issue_cycles'])==e-b
                assert abs(float(x['cycles_per_wave_group_load'])-(e-b)/n)<1e-8
                starts.append(min(int(x[f'wall_begin{w}']) for w in range(waves)));ends.append(max(int(x[f'wall_issue{w}']) for w in range(waves)));dones.append(max(int(x[f'wall_done{w}']) for w in range(waves)))
            tick_us=float(v[0]['issue_us'])/(ends[0]-starts[0]);grid_us=(max(ends)-min(starts))*tick_us;done_us=(max(dones)-min(starts))*tick_us
            payload=ctas*int(v[0]['requested_bytes']);bandwidth.append(payload/(grid_us*1000));done_bandwidth.append(payload/(done_us*1000));event_bw.append(payload/(float(v[0]['event_us'])*1000))
            periods.append(median(float(x['cycles_per_wave_group_load']) for x in v));overlaps.append(max(starts)<min(ends))
        if not smoke:
            row=dict(ctas=ctas,waves=waves,repeat=repeat,source_MiB=ctas,grid_GBps=median(bandwidth),completed_grid_GBps=median(done_bandwidth),event_GBps=median(event_bw),median_cta_cycles_per_step=median(periods),core_MHz=median(float(x['core_MHz']) for x in data),reported_resident_ctas_per_cu=int(re.search(r'resident_CTA_limit_per_CU=(\d+)',log)[1]),all_issue_intervals_overlap=all(overlaps),checked_cta_rows=len(data))
            rows.append(row)
            with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
            print(f"{stem}: {row['grid_GBps']:.1f} requested GB/s, all overlap={all(overlaps)}",flush=True)
    for waves in (1,2,4):run(2,waves,0,True)
    jobs=[(n,w) for n in counts for w in (1,2,4)]
    for repeat in (1,2):
        for ctas,waves in jobs if repeat==1 else reversed(jobs):run(ctas,waves,repeat)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(f"COMPLETE {len(rows)*3} checked timing launches,{sum(x['checked_cta_rows'] for x in rows)} CTA records,3 multi-CTA fill smokes; all ring words/guards pass.")
