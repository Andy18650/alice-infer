#!/usr/bin/env python3
"""Single-CTA cache/depth diagnostic; report body clocks, not pre-read event TF."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/latency')
audit=json.loads((folder/'audit.json').read_text())
metadata={(r['format'],r['depth']) for r in audit['kernels']}
assert len(metadata)==10
expected=metadata|{(-1,0)}
fields=('cycles_per_mfma','min_wave_cycles','max_wave_cycles','body_us_mean','core_MHz','launch_ms')
medians={}; out=[]
for run,rounds,steps in [('run1',9,4096),('run2',9,4096),('small',3,1024),('longer',3,8192)]:
    rows=list(csv.DictReader((folder/f'{run}.csv').open()))
    assert len(rows)==22*rounds
    log=(folder/f'{run}.log').read_text()
    assert 'timed_grid=1x1 block=256 waves=4' in log
    assert 'PASS eviction reader checksum controls' in log
    assert 'PASS 88 independent A/B changed-input/poisoned-output cases and guards' in log
    grouped=defaultdict(list); seen=set()
    for row in rows:
        fmt,depth,cache,round_=(int(row[k]) for k in ('format','depth','cache','round'))
        ident=fmt,depth,cache,round_
        assert ident not in seen; seen.add(ident)
        assert cache in (0,1) and int(row['steps'])==steps and int(row['eviction_MiB'])==256
        assert int(row['valid'])==1 and float(row['max_abs_error'])==0
        cycles=float(row['cycles_per_mfma'])
        assert float(row['min_wave_cycles'])<=cycles<=float(row['max_wave_cycles'])
        assert abs(cycles*steps/float(row['body_us_mean'])-float(row['core_MHz']))<0.002
        grouped[fmt,depth,cache].append(row)
    assert set(grouped)=={(f,d,c) for f,d in expected for c in (0,1)}
    for key,samples in sorted(grouped.items()):
        assert len(samples)==rounds
        values={f:median(float(r[f]) for r in samples) for f in fields}
        medians[run,*key]=values
        clocks=[float(r['cycles_per_mfma']) for r in samples]
        out.append([run,steps,*key,*(values[f] for f in fields),min(clocks),max(clocks)])
with (folder/'summary.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['run','steps','format','depth','cache',*fields,'min_sample_cycles','max_sample_cycles']); writer.writerows(out)
lines=['ONE timed CTA, four waves along N. format -1=resident MFMA, 1=FP8 B, 0=FP16 B.',
       'Warm = same-dispatch pre-read before clocks; evicted = completed 256 MiB reader, no pre-read.',
       'run | format | D | warm cycles/MFMA | evicted cycles/MFMA | warm MHz | evicted MHz']
for run in ('run1','run2','small','longer'):
    for fmt in (-1,1,0):
        for depth in ([0] if fmt<0 else [1,2,4,8,16]):
            warm=medians[run,fmt,depth,0]; evicted=medians[run,fmt,depth,1]
            lines.append(f"{run} | {fmt} | {depth} | {warm['cycles_per_mfma']:.3f} | {evicted['cycles_per_mfma']:.3f} | {warm['core_MHz']:.1f} | {evicted['core_MHz']:.1f}")
lines+=['','Small-footprint warm request-payload proxies (not physical cache bandwidth):']
for fmt in (1,0):
    cycles=medians['small',fmt,8,0]['cycles_per_mfma']
    payload=4*(512+(256 if fmt else 512))
    lines.append(f'format {fmt}, D8: {payload/cycles:.3f} logical A+B bytes per core cycle per CTA')
lines+=['','PASS 528 exact-valid timed samples and 352 independent warm/cold-path changed-input/poison/guard cases.',
        'Event launch_ms includes the warm pre-read: do not compare it as timed-body TFLOP/s.',
        'D16 does not close the >64 gap; these results alone do not uniquely identify its cause.',
        'No 2x2 wave-layout route or production change was pursued.']
text='\n'.join(lines)+'\n'; (folder/'summary.txt').write_text(text); print(text)
