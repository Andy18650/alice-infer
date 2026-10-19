#!/usr/bin/env python3
"""Summarize timings and independently reconstruct CTA spans from raw clocks.
No computed matrix values are read or checked.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median

p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='?',default=Path(__file__).resolve().parent/'results');a=p.parse_args()
groups=defaultdict(list);total=0
for path in sorted(a.results.glob('*/*.csv')):
    if path.name=='summary.csv': continue
    for row in csv.DictReader(path.open()):
        steps=int(row['steps'])
        starts=[int(row[f'begin{w}']) for w in range(4)]
        ends=[int(row[f'end{w}']) for w in range(4)]
        span=max(ends)-min(starts)
        assert all(e>b for b,e in zip(starts,ends))
        assert span==int(row['cta_core_cycles'])
        assert abs(span/steps-float(row['cta_cycles_per_step']))<1e-7
        assert span>=max(e-b for b,e in zip(starts,ends))
        assert abs(4096*steps/span-float(row['logical_B_per_core_cycle']))<1e-7
        key=(int(row['depth']),int(row['unroll']),row['mode'],int(row['loads']),int(row['wait']),int(row.get('lds_stride',1)),int(row.get('ready_banks',2)))
        run=path.stem.split('-run')[-1] if '-run' in path.stem else 'example'
        groups[key].append((run,row));total+=1
fields=('cta_cycles_per_step','mean_wave_cycles_per_step','logical_B_per_core_cycle','core_MHz','launch_us','body_us')
summary=[]
for key,samples in sorted(groups.items()):
    d,u,mode,loads,wait,stride,ready=key
    r=dict(depth=d,unroll=u,mode=mode,loads=loads,wait=wait,lds_stride=stride,ready_banks=ready,samples=len(samples),
           records=int(samples[0][1]['records']),active_bytes=int(samples[0][1]['active_bytes']),phases=int(samples[0][1]['phases']))
    r.update({f:median(float(row[f]) for _,row in samples) for f in fields})
    r.update({f'run{run}_cycles':median(float(row['cta_cycles_per_step']) for run_,row in samples if run_==run)
              if any(run_==run for run_,_ in samples) else '' for run in ('1','2')})
    summary.append(r)
with (a.results/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
lines=['Timing only: no numerical tests or output readback.',f'{total} sample CTA spans independently reconstructed from absolute wave clocks.',
       'depth/unroll | mode | loads | wait | LDS stride | CTA cycles/step run1/run2 (or example median)']
for r in summary:
    pair=f"{r['run1_cycles']:.3f}/{r['run2_cycles']:.3f}" if r['run1_cycles']!='' else f"{r['cta_cycles_per_step']:.3f}"
    lines.append(f"{r['depth']}/{r['unroll']} | {r['mode']} | {r['loads']} | {r['wait']} | {r['lds_stride']} | R{r['ready_banks']} | {pair}")
(a.results/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
