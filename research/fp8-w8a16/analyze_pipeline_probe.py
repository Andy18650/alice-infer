#!/usr/bin/env python3
"""Summarize 1/2/4/8-deep packed-weight pipelines with common input layout."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/pipeline')
labels=['resident_fp16','resident_fp8_8_8','stream_fp16','stream_fp8']
metadata={tuple(row[k] for k in ('mode','depth','unroll')):row for row in json.loads((folder/'audit.json').read_text())['kernels']}
assert len(metadata)==20
fields=('cycles_per_mfma','TFLOPs','useful_weight_GBs','requested_weight_GBs','core_MHz','ms')
medians={}; rows_out=[]
for run,rounds,steps in [('run1',9,4096),('run2',9,4096),('longer',3,8192)]:
    rows=list(csv.DictReader((folder/f'{run}.csv').open()))
    assert len(rows)==20*rounds
    assert 'PASS 40 changed-input/poisoned-output cases and output guards' in (folder/f'{run}.log').read_text()
    grouped=defaultdict(list); seen=set()
    for row in rows:
        key=tuple(int(row[k]) for k in ('mode','depth','unroll'))
        ident=(*key,int(row['round']))
        assert ident not in seen; seen.add(ident)
        assert int(row['steps'])==steps
        assert int(row['valid'])==1 and float(row['max_abs_error'])==0
        mode,depth,_=key
        if mode>=2:
            intensity=32 if mode==2 else 64
            assert abs(float(row['useful_weight_GBs'])-float(row['TFLOPs'])*1000/intensity)<0.001
            assert abs(float(row['requested_weight_GBs'])/float(row['useful_weight_GBs'])-(steps+depth+1)/steps)<1e-6
        else:
            assert float(row['useful_weight_GBs'])==float(row['requested_weight_GBs'])==0
        grouped[key].append(row)
    assert set(grouped)==set(metadata)
    for key,samples in sorted(grouped.items()):
        assert len(samples)==rounds
        values={f:median(float(row[f]) for row in samples) for f in fields}
        medians[run,*key]=values
        cycles=[float(row['cycles_per_mfma']) for row in samples]
        rows_out.append([run,steps,*key,*(values[f] for f in fields),min(cycles),max(cycles)])
with (folder/'summary.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['run','steps','mode','depth','unroll',*fields,'min_cycles','max_cycles'])
    writer.writerows(rows_out)
lines=['Rolling load-to-convert depth sweep: A register-resident, ordinary 8/8 body.',
       'run | mode | depth | U | cycles/MFMA | TFLOP/s | useful GB/s | MHz | TF speedup vs D1']
speedups=[]
for run in ('run1','run2','longer'):
    for mode in (3,2):
        for depth in (1,2,4,8):
            for unroll in (8,32):
                v=medians[run,mode,depth,unroll]
                speedup=v['TFLOPs']/medians[run,mode,1,unroll]['TFLOPs']
                speedups.append([run,mode,depth,unroll,speedup])
                lines.append(f"{run} | {labels[mode]} | {depth} | {unroll} | {v['cycles_per_mfma']:.3f} | {v['TFLOPs']:.3f} | {v['useful_weight_GBs']:.3f} | {v['core_MHz']:.1f} | {speedup:.4f}x")
with (folder/'speedups.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['run','mode','depth','unroll','event_TFLOPs_ratio_vs_depth1'])
    writer.writerows(speedups)
lines+=['','PASS 420 exact-valid timed samples and 120 changed-input/poisoned-output/guard cases.',
        'Depth 1 is the matched control in this common-stride harness, not the old allocation.',
        'No A-loading, weight-reuse, grid or backedge tuning; no fixed backedge fit under memory variability.']
text='\n'.join(lines)+'\n'
(folder/'summary.txt').write_text(text)
print(text)
