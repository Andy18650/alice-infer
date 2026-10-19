#!/usr/bin/env python3
"""Finite-K timing summaries only; no output readback or numerical tests."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

folder=Path(__file__).resolve().parent/'results/shallow-k'
groups=defaultdict(list)
for p in sorted(folder.glob('*-run[12].csv')):
    mode=p.name.split('-')[0]; run=p.stem.split('-')[-1]
    for row in csv.DictReader(p.open()):
        key=(mode,int(row['steps']),int(row['format']),int(row['depth']),int(row['cache']))
        groups[key].append((run,row))
summary=[]
for (mode,steps,fmt,depth,cache),samples in sorted(groups.items()):
    c=median(float(r['cycles_per_mfma']) for _,r in samples)
    slope=64 if fmt<0 else (96 if fmt else 128)
    rmed={run:median(float(r['cycles_per_mfma']) for run_,r in samples if run_==run) for run in ('run1','run2')}
    summary.append(dict(mode=mode,steps=steps,logical_K=steps*8,format=fmt,depth=depth,cache=cache,
                        samples=len(samples),cycles_per_mfma=c,run1_cycles=rmed['run1'],run2_cycles=rmed['run2'],
                        total_body_core_cycles=c*steps,residual_vs_long_slope=(c-slope)*steps,
                        body_us=median(float(r['body_us_mean']) for _,r in samples),
                        launch_us=median(float(r['launch_ms'])*1000 for _,r in samples),
                        core_MHz=median(float(r['core_MHz']) for _,r in samples)))
with (folder/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n'); w.writeheader(); w.writerows(summary)
index={(r['mode'],r['steps'],r['format'],r['depth'],r['cache']):r for r in summary}
text=['Performance-only, same previously measured executables. No numerical checks.',
      'D8, in-dispatch pre-read; median over two nine-round processes.',
      'steps | logical K | FP8 waited | FP8 no-wait | FP16 no-wait | FP8 no-wait residual cycles above 96*steps']
for s in (32,64,128,256,512,1024,4096):
    a=index['waited',s,1,8,0]; b=index['unwaited',s,1,8,0]; c=index['unwaited',s,0,8,0]
    text.append(f"{s} | {8*s} | {a['cycles_per_mfma']:.3f} | {b['cycles_per_mfma']:.3f} | {c['cycles_per_mfma']:.3f} | {b['residual_vs_long_slope']:.1f}")
text+=['','All depths, warm, unwaited. FP8 residual cycles above 96*steps:', 'steps | D1 | D2 | D4 | D8 | D16']
for s in (32,64,128,256,512,1024,4096):
    text.append(str(s)+' | '+' | '.join(f"{index['unwaited',s,1,d,0]['residual_vs_long_slope']:.1f}" for d in (1,2,4,8,16)))
text+=['','The long-loop throughput model needs a finite-length term.',
       'FP8 D8: T_body approximately 96*steps + 440..470 cycles for the pre-read path.',
       'Other variants have larger and less constant residuals; do not assign them all to one hardware latency.',
       'The original warm-up favors FP8 D8 and the resident anchor. Other cases are not individually prewarmed.',
       'Storage pitch and allocation sizes also vary with steps; addresses remain fixed within each timed loop.',
       'Timers exclude seed/queue-issue setup but include remaining in-flight work and the final drain.',
       'No claim that launch, cache/code-entry, fill and drain costs have been individually isolated.']
(folder/'summary.txt').write_text('\n'.join(text)+'\n'); print('\n'.join(text))
