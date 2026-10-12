#!/usr/bin/env python3
"""Keep cache regimes/protocols separate; compare hardware and dense references."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser()
p.add_argument('--roof-csv',required=True)
p.add_argument('--roof-log',required=True)
p.add_argument('--input',nargs='+',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
log=Path(a.roof_log).read_text()
by_grid=collections.defaultdict(list)
for blocks,tf in re.findall(r'MFMA_REFERENCE blocks=(\d+).*?tflops=([\d.]+)',log):
    by_grid[int(blocks)].append(float(tf))
if not by_grid: raise SystemExit('No arithmetic-only calibration in roof log')
peak=max(statistics.median(v) for v in by_grid.values())
bw=statistics.median(float(x) for x in re.findall(r'READ_REFERENCE.*?useful_GBps=([\d.]+)',log))
roof=collections.defaultdict(list)
for r in csv.DictReader(open(a.roof_csv)):
    if r['mode']=='graph' and int(r['repeats'])>1:
        roof[(r['m'],r['n'],r['k'],r['impl'])].append(float(r['tflops']))
dense=max(statistics.median(v) for v in roof.values())
references={'mfma_register_tflops':peak,'read_useful_GBps':bw,'dense_observed_tflops':dense,
            'warning':'Measured reference envelopes, not proven absolute hardware limits; read bytes are useful, not counters.'}
(out/'references.json').write_text(json.dumps(references,indent=2)+'\n')
keys=['m','n','k','axis','q','streams','impl','cache','mode','k_launch','bpad','weights','repeats']
groups=collections.defaultdict(list)
for file in a.input:
    for r in csv.DictReader(open(file)):
        r.setdefault('k_launch','separate');r.setdefault('bpad','0');r.setdefault('weights','column')
        groups[tuple(r[k] for k in keys)].append(r)
summary=[]
for key,rows in groups.items():
    r=rows[0];m,n,k,q=map(int,(r['m'],r['n'],r['k'],r['q']))
    flops=2*m*n*k*q;us=statistics.median(float(x['gpu_us']) for x in rows)
    tf=flops/us/1e6;intensity=flops/float(r['compulsory_bytes']);modeled=flops/float(r['modeled_bytes'])
    hardware_roof=min(peak,bw*intensity/1000)
    algorithm_roof=min(peak,bw*modeled/1000)
    dense_roof=min(dense,bw*modeled/1000)
    weight_bytes=2*n*k*(1 if r['axis']=='m' else q)
    s=dict(zip(keys,key));s.update(samples=len(rows),gpu_us=us,weight_GBps=weight_bytes/us/1000,
      cpu_us=statistics.median(float(x['cpu_us']) for x in rows),tflops=tf,
      intensity=intensity,hardware_roof_tflops=hardware_roof,
      hardware_efficiency=tf/hardware_roof,modeled_efficiency=tf/algorithm_roof,
      dense_reference_efficiency=tf/dense_roof,ctas=r['ctas'],registers=r['registers'],
      lds_bytes=r['lds_bytes'],scratch_bytes=r['scratch_bytes'],blocks_per_cu=r['blocks_per_cu'])
    summary.append(s)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
# Winner selection is within one shape and one protocol, never across protocols.
best={}
for s in summary:
    key=tuple(s[k] for k in keys if k!='impl')
    if key not in best or s['gpu_us']<best[key]['gpu_us']:best[key]=s
with (out/'best.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(best.values())
# 3D Pareto frontier is only meaningful at fixed q/axis/cache/mode/repetitions.
# Warm points are NOT presented as HBM roofline measurements.
frontiers=[]
for metric in ['hardware_efficiency','dense_reference_efficiency']:
    for threshold in [.90,.95]:
        buckets=collections.defaultdict(list)
        for s in best.values():
            if s[metric]>=threshold:
                buckets[tuple(s[k] for k in ['axis','q','streams','cache','mode','k_launch','bpad','weights','repeats'])].append(s)
        for bucket,points in buckets.items():
            for s in points:
                dims=tuple(int(s[k]) for k in ['m','n','k'])
                if any(all(int(t[k])<=v for k,v in zip(['m','n','k'],dims)) and
                       any(int(t[k])<v for k,v in zip(['m','n','k'],dims)) for t in points):continue
                frontiers.append(dict(s,metric=metric,threshold=threshold))
if frontiers:
    with (out/'frontier.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(frontiers[0]),lineterminator='\n');w.writeheader();w.writerows(frontiers)
print(json.dumps(references,indent=2))
print(f'{len(summary)} configuration medians, {len(best)} shape/protocol winners, {len(frontiers)} frontier points')
