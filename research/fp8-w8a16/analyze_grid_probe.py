#!/usr/bin/env python3
"""Compare equal-work grids; distinguish shader requests from unique inputs."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/grid')
metadata={tuple(r[k] for k in ('format','depth','unroll')):r for r in json.loads((folder/'audit.json').read_text())['kernels']}
assert len(metadata)==16
fields=('cycles_per_mfma','TFLOPs','useful_B_GBs','useful_A_request_GBs','unique_B_equiv_GBs','unique_A_equiv_GBs','requested_B_GBs','requested_A_GBs','core_MHz','ms')
medians={}; out=[]
for run,rounds,steps in [('run1',9,4096),('run2',9,4096),('longer',3,8192)]:
    rows=list(csv.DictReader((folder/f'{run}.csv').open()))
    assert len(rows)==32*rounds
    assert 'PASS 128 independent A/B changed-input/poisoned-output cases and guards' in (folder/f'{run}.log').read_text()
    grouped=defaultdict(list); seen=set()
    for row in rows:
        key=tuple(int(row[k]) for k in ('format','grid_m','depth','unroll'))
        fmt,gm,d,u=key
        assert gm in (1,8) and int(row['grid_n'])*gm==104
        ident=(*key,int(row['round']))
        assert ident not in seen; seen.add(ident)
        assert int(row['steps'])==steps and int(row['valid'])==1 and float(row['max_abs_error'])==0
        b=float(row['useful_B_GBs']); a=float(row['useful_A_request_GBs'])
        assert abs(b-float(row['TFLOPs'])*1000/(64 if fmt else 32))<0.001
        assert abs(a-b*(2 if fmt else 1))<0.001
        assert abs(float(row['unique_B_equiv_GBs'])-b/gm)<0.001
        assert abs(float(row['unique_A_equiv_GBs'])-a*gm/416)<0.001
        assert abs(float(row['requested_B_GBs'])/b-(steps+d+1)/steps)<1e-6
        assert abs(float(row['requested_A_GBs'])/a-(steps+d+1)/steps)<1e-6
        grouped[key].append(row)
    assert set(grouped)=={(f,m,d,u) for f,d,u in metadata for m in (1,8)}
    for key,samples in sorted(grouped.items()):
        assert len(samples)==rounds
        values={f:median(float(row[f]) for row in samples) for f in fields}
        medians[run,*key]=values
        out.append([run,steps,*key,*(values[f] for f in fields),min(float(r['cycles_per_mfma']) for r in samples),max(float(r['cycles_per_mfma']) for r in samples)])
with (folder/'summary.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['run','steps','format','grid_m','depth','unroll',*fields,'min_cycles','max_cycles']); w.writerows(out)
lines=['Equal-work grid comparison: x=N CTAs, y=M CTAs; four waves along N.',
       'run | format | D | U | 104x1 TF | 13x8 TF | speedup | old cycles | new cycles | new A+B request GB/s']
ratios=[]
for run in ('run1','run2','longer'):
    for fmt in (1,0):
        for d in (1,2,4,8):
            for u in (8,32):
                old=medians[run,fmt,1,d,u]; new=medians[run,fmt,8,d,u]
                ratio=new['TFLOPs']/old['TFLOPs']
                ratios.append([run,fmt,d,u,ratio,new['cycles_per_mfma']/old['cycles_per_mfma']])
                lines.append(f"{run} | {fmt} | {d} | {u} | {old['TFLOPs']:.3f} | {new['TFLOPs']:.3f} | {ratio:.4f}x | {old['cycles_per_mfma']:.3f} | {new['cycles_per_mfma']:.3f} | {new['useful_B_GBs']+new['useful_A_request_GBs']:.3f}")
with (folder/'comparisons.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['run','format','depth','unroll','TFLOPs_ratio_13x8_over_104x1','cycles_ratio_13x8_over_104x1']); w.writerows(ratios)
lines+=['','D8/U8 bandwidth details (GB/s): run | format | gridM | B requests | A requests | unique B equivalent | unique A equivalent']
for run in ('run1','run2','longer'):
    for fmt in (1,0):
        for gm in (1,8):
            v=medians[run,fmt,gm,8,8]
            lines.append(f"{run} | {fmt} | {gm} | {v['useful_B_GBs']:.3f} | {v['useful_A_request_GBs']:.3f} | {v['unique_B_equiv_GBs']:.3f} | {v['unique_A_equiv_GBs']:.3f}")
lines+=['','PASS 672 exact-valid timed samples and 384 independent A/B-change/poison/guard cases.',
        'Both grids launch the same compiled entry per format/depth/unroll; all 16 entries audited.',
        'Shader-request and unique-matrix-equivalent rates are NOT physical HBM bandwidth.',
        'M/N shapes and input sharing change at equal FLOPs; no unique NoC attribution or 2x2 wave tiling.']
text='\n'.join(lines)+'\n'; (folder/'summary.txt').write_text(text); print(text)
