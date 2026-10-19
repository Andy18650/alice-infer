#!/usr/bin/env python3
"""Compare timing-only reruns after stopping the idle llama server."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

root=Path(__file__).resolve().parent/'results'
groups=defaultdict(list)
for state,folder in [('server-resident','shallow-k'),('server-stopped','server-off')]:
    for p in (root/folder).glob('*-run[12].csv'):
        mode=p.name.split('-')[0]
        for r in csv.DictReader(p.open()):
            steps=int(r['steps'])
            if steps not in (32,4096): continue
            key=mode,steps,int(r['format']),int(r['depth']),int(r['cache'])
            groups[state,key].append(float(r['cycles_per_mfma']))
rows=[]
for key in sorted(k for state,k in groups if state=='server-stopped'):
    before=median(groups['server-resident',key]); after=median(groups['server-stopped',key])
    mode,steps,fmt,depth,cache=key
    rows.append(dict(mode=mode,steps=steps,logical_K=steps*8,format=fmt,depth=depth,cache=cache,
                     resident_cycles=before,stopped_cycles=after,change_percent=100*(after/before-1)))
folder=root/'server-off'
with (folder/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
lines=['Timing-only rerun; same binaries, no numerical tests.',
       'mode | steps | format | depth | cache | resident cycles | stopped cycles | change %']
for r in rows:
    lines.append(f"{r['mode']} | {r['steps']} | {r['format']} | {r['depth']} | {r['cache']} | {r['resident_cycles']:.3f} | {r['stopped_cycles']:.3f} | {r['change_percent']:+.2f}%")
(folder/'comparison.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
