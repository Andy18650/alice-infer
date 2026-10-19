#!/usr/bin/env python3
"""Timing summaries only: no numerical output validation or correctness gates."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

folder=Path(__file__).resolve().parent/'results/wait'
summary=[]
for run in ('run1','run2','longer'):
    for mode in ('waited','unwaited'):
        groups=defaultdict(list)
        for row in csv.DictReader((folder/f'{mode}-{run}.csv').open()):
            groups[int(row['format']),int(row['depth']),int(row['cache'])].append(row)
        for (fmt,depth,cache),rows in sorted(groups.items()):
            c=median(float(r['cycles_per_mfma']) for r in rows)
            summary.append(dict(run=run,mode=mode,format=fmt,depth=depth,cache=cache,
                                samples=len(rows),steps=int(rows[0]['steps']),cycles_per_mfma=c,
                                core_MHz=median(float(r['core_MHz']) for r in rows)))
with (folder/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n'); w.writeheader(); w.writerows(summary)
text=['Performance only. No output readback, references, numerical tests or validity fields.',
      'run | mode | B format | D | cache | median core cycles/MFMA']
for r in summary:
    text.append(f"{r['run']} | {r['mode']} | {r['format']} | {r['depth']} | {r['cache']} | {r['cycles_per_mfma']:.3f}")
text+='''
Working throughput model, not identification of a physical hardware port:
T_step ~= max(64 MFMA cycles, four-wave logical VMEM payload / R_effective).
R_effective ~= 32 logical bytes/core-cycle, fitted from the deep FP8 case.
FP8 payload 3072 B -> 96 cycles; FP16 payload 4096 B -> 128 cycles.
Removing explicit waits removes shallow-depth penalties, not this deep floor.
No claim of valid math is made for the unwaited kernel.
'''.splitlines()
(folder/'summary.txt').write_text('\n'.join(text)+'\n')
print('\n'.join(text))
