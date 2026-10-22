#!/usr/bin/env python3
"""Matched64-CTA disjoint/8x8 source-sharing A/B, GPU bases verified."""
import argparse,collections,csv,json,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/broadcast-vgpr');p.add_argument('--output',type=Path,default=here/'results/broadcast-vgpr');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve()
if out.exists() and any(out.iterdir()):p.error('choose a fresh results directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw','smoke'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('broadcast-vgpr.hip','generate-vgpr.py','build-broadcast-vgpr.sh','inspect_vgpr.py','run_broadcast_vgpr.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('vgpr-loads.hpp','compiler-flags.txt','hashes.txt','metadata.txt','loop.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),ctas=64,waves=4,ring_slots=32,requested_bytes_per_cta=268435456,allocated_source_bytes=67108864,reachable_source_bytes=dict(disjoint=67108864,outer=8388608),placement='numactl CPU+memory node0'),indent=2)+'\n')
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
rows=[]
with (out/'commands.txt').open('w') as commands:
    def run(layout,repeat,groups=None):
        smoke=groups is not None;folder='smoke' if smoke else 'raw';stem=f'{layout}-r{repeat}'+(f'-g{groups}' if smoke else '')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--layout',layout,'--rounds','1' if smoke else '9','--warmup-ms','0' if smoke else '200']
        if smoke:cmd+=['--groups',str(groups)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        with (out/folder/(stem+'.csv')).open('w') as f,(out/folder/(stem+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with (out/folder/(stem+'.csv')).open() as f:data=list(csv.DictReader(f))
        assert len(data)==64*(1 if smoke else 9)
        assert 'GPU source-base map exact' in (out/folder/(stem+'.log')).read_text()
        bw=[];done_bw=[];events=[];periods=[];min_cta=[];max_cta=[];overlaps=[]
        for r in range(1 if smoke else 9):
            v=[x for x in data if int(x['round'])==r];assert {int(x['cta']) for x in v}==set(range(64))
            bases=collections.Counter();starts=[];ends=[];dones=[]
            for x in v:
                cta=int(x['cta']);assert x['layout']==layout and int(x['ctas'])==64 and int(x['waves'])==4
                assert x['valid']=='1' and x['mismatches']==x['guards']=='0'
                assert int(x['ring_slots'])==32 and int(x['source_bytes_per_cta'])==1048576
                n=int(x['groups'])*32;assert int(x['loads_per_wave'])==n and int(x['requested_bytes'])==n*4096
                if not smoke:assert int(x['requested_bytes'])==268435456
                b=min(int(x[f'begin{w}']) for w in range(4));e=max(int(x[f'issue{w}']) for w in range(4));assert e-b==int(x['issue_cycles'])
                assert abs(float(x['cycles_per_wave_group_load'])-(e-b)/n)<1e-8
                for w in range(4):
                    expected=(cta//8)*524288+w*262144 if w<2 else 4194304+(cta%8)*524288+(w%2)*262144
                    if layout=='disjoint':expected=cta*1048576+w*262144
                    actual=int(x[f'source_offset{w}']);assert actual==expected;bases[actual]+=1
                starts.append(min(int(x[f'wall_begin{w}']) for w in range(4)));ends.append(max(int(x[f'wall_issue{w}']) for w in range(4)));dones.append(max(int(x[f'wall_done{w}']) for w in range(4)))
            assert len(bases)==(32 if layout=='outer' else 256)
            assert set(bases.values())==({8} if layout=='outer' else {1})
            tick_us=float(v[0]['issue_us'])/(ends[0]-starts[0]);grid_us=(max(ends)-min(starts))*tick_us;done_us=(max(dones)-min(starts))*tick_us
            payload=64*int(v[0]['requested_bytes']);bw.append(payload/(grid_us*1000));done_bw.append(payload/(done_us*1000));events.append(payload/(float(v[0]['event_us'])*1000))
            values=[float(x['cycles_per_wave_group_load']) for x in v];periods.append(median(values));min_cta.append(min(values));max_cta.append(max(values));overlaps.append(max(starts)<min(ends))
        assert all(overlaps)
        if not smoke:
            row=dict(layout=layout,repeat=repeat,source_MiB=8 if layout=='outer' else 64,allocated_source_MiB=64,grid_GBps=median(bw),completion_GBps=median(done_bw),event_GBps=median(events),cta_cycles_per_step=median(periods),min_cta_cycles=median(min_cta),max_cta_cycles=median(max_cta),core_MHz=median(float(x['core_MHz']) for x in data),grid_timings=9,checked_cta_rows=len(data),all_overlap=True,base_reuse=8 if layout=='outer' else 1)
            rows.append(row)
            with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
            print(row,flush=True)
    for layout in ('disjoint','outer'):
        for groups in (1,9):run(layout,0,groups)
    for repeat,order in ((1,('disjoint','outer')),(2,('outer','disjoint'))):
        for layout in order:run(layout,repeat)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('COMPLETE36 checked grid timings/2304 CTA records +4 fill/wrap grid smokes; all ring words/guards and256 GPU source bases/launch verified.')
