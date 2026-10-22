#!/usr/bin/env python3
"""Requested panel-sharing rectangles plus8x8 control; same native loop."""
import argparse,collections,csv,json,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/broadcast-shapes');p.add_argument('--output',type=Path,default=here/'results/broadcast-shapes');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve();shapes=((4,4),(4,8),(8,8),(13,8))
if out.exists() and any(out.iterdir()):p.error('choose a fresh results directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw','smoke'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('broadcast-shapes.hip','generate-vgpr.py','build-broadcast-shapes.sh','inspect_vgpr.py','run_broadcast_shapes.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('vgpr-loads.hpp','compiler-flags.txt','hashes.txt','metadata.txt','loop.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),shapes=shapes,waves=4,ring_slots=32,panel_bytes=524288,requested_bytes_per_cta=268435456,placement='numactl CPU+memory node0'),indent=2)+'\n')
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
summary=[]
with (out/'commands.txt').open('w') as commands:
    def run(rows,cols,layout,repeat,groups=None):
        ctas=rows*cols;smoke=groups is not None;folder='smoke' if smoke else 'raw';stem=f'{rows}x{cols}-{layout}-r{repeat}'+(f'-g{groups}' if smoke else '')
        cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--rows',str(rows),'--cols',str(cols),'--layout',layout,'--rounds','1' if smoke else '9','--warmup-ms','0' if smoke else '200']
        if smoke:cmd+=['--groups',str(groups)]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        with (out/folder/(stem+'.csv')).open('w') as f,(out/folder/(stem+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
        with (out/folder/(stem+'.csv')).open() as f:data=list(csv.DictReader(f))
        assert len(data)==ctas*(1 if smoke else 9)
        assert 'GPU source-base map exact' in (out/folder/(stem+'.log')).read_text()
        bw=[];complete_bw=[];event_bw=[];period=[];overlap=[]
        for r in range(1 if smoke else 9):
            v=[x for x in data if int(x['round'])==r];assert {int(x['cta']) for x in v}==set(range(ctas))
            bases=collections.Counter();a_bases=collections.Counter();b_bases=collections.Counter();starts=[];ends=[];dones=[]
            for x in v:
                c=int(x['cta']);assert int(x['rows'])==rows and int(x['cols'])==cols and int(x['ctas'])==ctas and int(x['waves'])==4 and x['layout']==layout
                assert x['valid']=='1' and x['mismatches']==x['guards']=='0'
                n=int(x['groups'])*32;assert int(x['loads_per_wave'])==n and int(x['requested_bytes'])==n*4096
                if not smoke:assert int(x['requested_bytes'])==268435456
                b=min(int(x[f'begin{w}']) for w in range(4));e=max(int(x[f'issue{w}']) for w in range(4));assert e-b==int(x['issue_cycles'])
                assert abs(float(x['cycles_per_wave_group_load'])-(e-b)/n)<1e-8
                for w in range(4):
                    expected=(c//cols)*524288+(w%2)*262144 if w<2 else rows*524288+(c%cols)*524288+(w%2)*262144
                    if layout=='disjoint':expected=c*1048576+w*262144
                    actual=int(x[f'source_offset{w}']);assert actual==expected;bases[actual]+=1
                    (a_bases if w<2 else b_bases)[actual]+=1
                starts.append(min(int(x[f'wall_begin{w}']) for w in range(4)));ends.append(max(int(x[f'wall_issue{w}']) for w in range(4)));dones.append(max(int(x[f'wall_done{w}']) for w in range(4)))
            if layout=='outer':
                assert len(a_bases)==2*rows and set(a_bases.values())=={cols}
                assert len(b_bases)==2*cols and set(b_bases.values())=={rows}
                assert not set(a_bases).intersection(b_bases)
            else:assert len(bases)==4*ctas and set(bases.values())=={1}
            tick_us=float(v[0]['issue_us'])/(ends[0]-starts[0]);grid_us=(max(ends)-min(starts))*tick_us;done_us=(max(dones)-min(starts))*tick_us
            payload=ctas*int(v[0]['requested_bytes']);bw.append(payload/(grid_us*1000));complete_bw.append(payload/(done_us*1000));event_bw.append(payload/(float(v[0]['event_us'])*1000))
            period.append(median(float(x['cycles_per_wave_group_load']) for x in v));overlap.append(max(starts)<min(ends))
        assert all(overlap)
        if not smoke:
            row=dict(rows=rows,cols=cols,ctas=ctas,layout=layout,repeat=repeat,active_source_MiB=(rows+cols)/2 if layout=='outer' else ctas,allocated_source_MiB=ctas,grid_GBps=median(bw),completion_GBps=median(complete_bw),event_GBps=median(event_bw),cta_cycles_per_step=median(period),core_MHz=median(float(x['core_MHz']) for x in data),checked_cta_records=len(data),all_overlap=True)
            summary.append(row)
            with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
            print(row,flush=True)
    for rows,cols in shapes:
        for layout in ('disjoint','outer'):
            for groups in (1,9):run(rows,cols,layout,0,groups)
    jobs=[(r,c,l) for r,c in shapes for l in ('disjoint','outer')]
    for repeat in (1,2):
        for rows,cols,layout in jobs if repeat==1 else reversed(jobs):run(rows,cols,layout,repeat)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(f"COMPLETE {len(summary)*9} checked grid timings/{sum(x['checked_cta_records'] for x in summary)} CTA records +16 fill/wrap grid smokes; all ring words/guards/GPU source maps pass.")
