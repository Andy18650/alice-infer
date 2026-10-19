#!/usr/bin/env python3
"""Compare repeated fragment-zero loads with the retained real-K controls."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/fixed')
audit=json.loads((folder/'audit.json').read_text())
expected={(r['format'],r['fixed_k'],r['depth']) for r in audit['kernels']}|{(-1,1,0)}
assert len(expected)==21
fields=('cycles_per_mfma','min_wave_cycles','max_wave_cycles','body_us_mean','core_MHz','launch_ms')
medians={}; out=[]
for run,rounds,steps in [('run1',9,4096),('run2',9,4096),('small',3,1024),('longer',3,8192)]:
    rows=list(csv.DictReader((folder/f'{run}.csv').open()))
    assert len(rows)==42*rounds
    log=(folder/f'{run}.log').read_text()
    assert 'timed_grid=1x1 block=256 waves=4' in log
    assert 'PASS eviction reader checksum controls' in log
    assert 'PASS 168 independent A/B changed-input/poisoned-output cases and guards' in log
    grouped=defaultdict(list); seen=set()
    for row in rows:
        fmt,fixed,depth,cache,round_=(int(row[k]) for k in ('format','fixed_k','depth','cache','round'))
        ident=fmt,fixed,depth,cache,round_
        assert ident not in seen; seen.add(ident)
        assert int(row['steps'])==steps and cache in (0,1) and int(row['eviction_MiB'])==256
        assert int(row['valid'])==1 and float(row['max_abs_error'])==0
        cycles=float(row['cycles_per_mfma'])
        assert float(row['min_wave_cycles'])<=cycles<=float(row['max_wave_cycles'])
        assert abs(cycles*steps/float(row['body_us_mean'])-float(row['core_MHz']))<0.002
        grouped[fmt,fixed,depth,cache].append(row)
    assert set(grouped)=={(f,k,d,c) for f,k,d in expected for c in (0,1)}
    for key,samples in sorted(grouped.items()):
        assert len(samples)==rounds
        values={f:median(float(r[f]) for r in samples) for f in fields}
        medians[run,*key]=values
        cs=[float(r['cycles_per_mfma']) for r in samples]
        out.append([run,steps,*key,*(values[f] for f in fields),min(cs),max(cs)])
with (folder/'summary.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['run','steps','format','fixed_k','depth','cache',*fields,'min_sample_cycles','max_sample_cycles']); w.writerows(out)
lines=['One CTA; all A/B loads retained, fixed K means first eight-column/row fragment.',
       'run | format | D | fixed warm cycles | fixed evict cycles | variable warm cycles | variable evict cycles']
for run in ('run1','run2','small','longer'):
    for fmt in (1,0):
        for d in (1,2,4,8,16):
            vals=[medians[run,fmt,k,d,c]['cycles_per_mfma'] for k,c in [(1,0),(1,1),(0,0),(0,1)]]
            lines.append(f'{run} | {fmt} | {d} | '+' | '.join(f'{v:.3f}' for v in vals))
lines+=['','Fixed D8 logical A+B request payload per mean CTA core cycle (not physical cache bandwidth):']
for run in ('run1','run2','longer'):
    for fmt in (1,0):
        payload=4*(512+(256 if fmt else 512))
        c=medians[run,fmt,1,8,0]['cycles_per_mfma']
        lines.append(f'{run}, format {fmt}: {payload/c:.3f} B/core-cycle, resident anchor {medians[run,-1,1,0,0]["cycles_per_mfma"]:.3f} cycles/MFMA')
lines+=['','PASS 1008 exact-valid timed samples and 672 independent A/B-change/warm-path/poison/guard cases.',
        'Fixed data alone cannot reveal stale same-value reads; retain FIFO safety audits and varying-K controls.',
        'K-dependent address arithmetic also folds away: this is not an isolated cache-latency subtraction.',
        'No counters, 2x2 tiling, load elimination or production changes.']
text='\n'.join(lines)+'\n'; (folder/'summary.txt').write_text(text); print(text)
