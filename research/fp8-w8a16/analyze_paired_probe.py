#!/usr/bin/env python3
"""Summarize shared real-A paired pipelines against co-run free-A controls."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/paired')
metadata={tuple(row[k] for k in ('format','stream_a','depth','unroll')):row for row in json.loads((folder/'audit.json').read_text())['kernels']}
assert len(metadata)==32
fields=('cycles_per_mfma','TFLOPs','useful_B_GBs','useful_A_request_GBs','requested_B_GBs','requested_A_GBs','core_MHz','ms')
medians={}; out=[]
for run,rounds,steps in [('run1',9,4096),('run2',9,4096),('longer',3,8192)]:
    rows=list(csv.DictReader((folder/f'{run}.csv').open()))
    assert len(rows)==32*rounds
    assert 'PASS 128 independent A/B changed-input/poisoned-output cases and guards' in (folder/f'{run}.log').read_text()
    grouped=defaultdict(list); seen=set()
    for row in rows:
        key=tuple(int(row[k]) for k in ('format','stream_a','depth','unroll'))
        ident=(*key,int(row['round']))
        assert ident not in seen; seen.add(ident)
        assert int(row['steps'])==steps
        assert int(row['valid'])==1 and float(row['max_abs_error'])==0
        fmt,streama,depth,_=key
        intensity=64 if fmt else 32
        b=float(row['useful_B_GBs']); a=float(row['useful_A_request_GBs'])
        assert abs(b-float(row['TFLOPs'])*1000/intensity)<0.001
        assert abs(float(row['requested_B_GBs'])/b-(steps+depth+1)/steps)<1e-6
        if streama:
            assert abs(a-b*(2 if fmt else 1))<0.001
            assert abs(float(row['requested_A_GBs'])/a-(steps+depth+1)/steps)<1e-6
        else:
            assert a==float(row['requested_A_GBs'])==0
        grouped[key].append(row)
    assert set(grouped)==set(metadata)
    for key,samples in sorted(grouped.items()):
        assert len(samples)==rounds
        values={f:median(float(row[f]) for row in samples) for f in fields}
        medians[run,*key]=values
        out.append([run,steps,*key,*(values[f] for f in fields),min(float(row['cycles_per_mfma']) for row in samples),max(float(row['cycles_per_mfma']) for row in samples)])
with (folder/'summary.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['run','steps','format','stream_a','depth','unroll',*fields,'min_cycles','max_cycles'])
    writer.writerows(out)
ratios=[]
lines=['Real MFMA-packed A, paired waits; format 1=FP8 B, format 0=FP16 B.',
       'run | format | D | U | free-A TF | real-A TF | real/free | free cycles | real cycles | B GB/s | A request GB/s']
for run in ('run1','run2','longer'):
    for fmt in (1,0):
        for depth in (1,2,4,8):
            for u in (8,32):
                free=medians[run,fmt,0,depth,u]; real=medians[run,fmt,1,depth,u]
                ratio=real['TFLOPs']/free['TFLOPs']
                ratios.append([run,fmt,depth,u,ratio,real['cycles_per_mfma']/free['cycles_per_mfma'],real['core_MHz']/free['core_MHz']])
                lines.append(f"{run} | {fmt} | {depth} | {u} | {free['TFLOPs']:.3f} | {real['TFLOPs']:.3f} | {ratio*100:.3f}% | {free['cycles_per_mfma']:.3f} | {real['cycles_per_mfma']:.3f} | {real['useful_B_GBs']:.3f} | {real['useful_A_request_GBs']:.3f}")
with (folder/'comparisons.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['run','format','depth','unroll','real_A_TFLOPs_over_free_A','real_A_cycles_over_free_A','real_A_core_MHz_over_free_A'])
    writer.writerows(ratios)
lines+=['','PASS all 672 timed samples exact; 384 independent A/B-change/poison/guard validations passed.',
        'A request bandwidth counts all wave requests to shared addresses, NOT physical HBM traffic.',
        'No A cache-hit ratio, standalone latency, grid/reuse/backedge tuning or compute-bound claim.']
text='\n'.join(lines)+'\n'
(folder/'summary.txt').write_text(text)
print(text)
