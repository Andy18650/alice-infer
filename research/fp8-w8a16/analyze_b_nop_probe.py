#!/usr/bin/env python3
"""Timing-only MFMA-to-NOP hypothesis test; no numerical output checks."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

folder=Path(__file__).resolve().parent/'results/b-nop'
groups=defaultdict(list)
for p in folder.glob('*-run[12].csv'):
    kind,mode,run=p.stem.split('-')
    for row in csv.DictReader(p.open()):
        groups[kind,mode,int(row['format']),int(row['depth']),int(row['cache'])].append((run,row))
def cycles(row): return float(row.get('cycles_per_step',row.get('cycles_per_mfma')))
rows=[]
for (kind,mode,fmt,depth,cache),samples in sorted(groups.items()):
    byrun={run:median(cycles(r) for run_,r in samples if run_==run) for run in ('run1','run2')}
    rows.append(dict(kind=kind,mode=mode,format=fmt,depth=depth,cache=cache,
                     steps=4096,logical_K=32768,samples=len(samples),
                     cycles_per_step=median(cycles(r) for _,r in samples),
                     run1_cycles=byrun['run1'],run2_cycles=byrun['run2'],
                     core_MHz=median(float(r['core_MHz']) for _,r in samples)))
with (folder/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
idx={(r['kind'],r['mode'],r['format'],r['depth'],r['cache']):r for r in rows}
lines=['Performance only, no numerical tests. MFMA -> s_nop 0; B preparation retained.',
       '4096 steps, logical K32768, four N waves. Pre-read preparation.',
       'mode | format | D | MFMA run1/run2 | NOP run1/run2']
for mode in ('unwaited','waited'):
    for fmt in (1,0):
        for d in (1,2,4,8,16):
            a=idx['mfma',mode,fmt,d,0]; b=idx['nop',mode,fmt,d,0]
            lines.append(f"{mode} | {fmt} | {d} | {a['run1_cycles']:.3f}/{a['run2_cycles']:.3f} | {b['run1_cycles']:.3f}/{b['run2_cycles']:.3f}")
lines+=['','D8 NOP no-wait, both preparations:']
for fmt in (1,0):
    for cache in (0,1):
        r=idx['nop','unwaited',fmt,8,cache]
        lines.append(f"format={fmt} cache={cache}: {r['cycles_per_step']:.3f} cycles/step")
lines+=['','Hypothesis verdict: partial match, NOT a validation of the complete payload-only model.',
        'FP16 ~64.6 is consistent with 2048 logical bytes / 32 B/core-cycle.',
        'FP8 ~45.39 does not reach the payload-only 1024/32 = 32 cycle prediction.',
        'The eight retained FP8 VALUs, NOP, issue/dependency/other effects are not isolated here.',
        'Do not infer a physical memory limit, or attribute the entire FP8 excess solely to conversion.',
        'No extra ablation is performed. Other-depth warm-up caveats remain.']
(folder/'summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
