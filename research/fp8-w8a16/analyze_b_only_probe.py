#!/usr/bin/env python3
"""Timing-only comparison of A+B loads versus resident-A/B-only loads."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

folder=Path(__file__).resolve().parent/'results/b-only'
groups=defaultdict(list)
for p in folder.glob('*-run[12].csv'):
    kind,mode,run=p.stem.split('-')
    for row in csv.DictReader(p.open()):
        groups[kind,mode,int(row['format']),int(row['depth']),int(row['cache'])].append((run,row))
rows=[]
for (kind,mode,fmt,depth,cache),samples in sorted(groups.items()):
    byrun={run:median(float(r['cycles_per_mfma']) for run_,r in samples if run_==run) for run in ('run1','run2')}
    rows.append(dict(kind=kind,mode=mode,format=fmt,depth=depth,cache=cache,
                     steps=4096,logical_K=32768,samples=len(samples),
                     cycles_per_mfma=median(float(r['cycles_per_mfma']) for _,r in samples),
                     run1_cycles=byrun['run1'],run2_cycles=byrun['run2'],
                     core_MHz=median(float(r['core_MHz']) for _,r in samples)))
with (folder/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
idx={(r['kind'],r['mode'],r['format'],r['depth'],r['cache']):r for r in rows}
lines=['Performance only: no numerical checks or output readback.',
       '4096 MFMA steps, logical K32768, four N waves. Pre-read preparation.',
       'mode | format | D | A+B run1/run2 | B-only run1/run2']
for mode in ('unwaited','waited'):
    for fmt in (1,0):
        for d in (1,2,4,8,16):
            a=idx['ab',mode,fmt,d,0]; b=idx['b',mode,fmt,d,0]
            lines.append(f"{mode} | {fmt} | {d} | {a['run1_cycles']:.3f}/{a['run2_cycles']:.3f} | {b['run1_cycles']:.3f}/{b['run2_cycles']:.3f}")
lines+=['','D8 no-wait pooled medians for both preparations:']
for fmt in (1,0):
    for cache in (0,1):
        a=idx['ab','unwaited',fmt,8,cache]; b=idx['b','unwaited',fmt,8,cache]
        lines.append(f"format={fmt} cache={cache}: A+B {a['cycles_per_mfma']:.3f}, B-only {b['cycles_per_mfma']:.3f}")
lines+=['','Working logical-payload model at R_effective ~= 32 B/core-cycle:',
        'A+B FP8: 3072 B -> max(64,96) = 96 cycles.',
        'A+B FP16: 4096 B -> max(64,128) = 128 cycles.',
        'B-only FP8: 1024 B -> max(64,32) = 64 cycles.',
        'B-only FP16: 2048 B -> max(64,64) = 64 cycles.',
        'A loads AND two A-copy VALUs disappear; this does not isolate their individual costs.',
        'Register allocation/code layout also change. No physical port/bandwidth-counter claim.',
        'Inherited warm-up favors FP8 D8 and the resident anchor; other-depth finite costs remain.']
(folder/'summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
