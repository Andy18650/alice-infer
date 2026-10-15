#!/usr/bin/env python3
"""Same-geometry layout ratios separately from per-layout tuned winners."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
p=argparse.ArgumentParser();p.add_argument('--input',nargs='+',required=True)
p.add_argument('--selection',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
groups=collections.defaultdict(list)
for file in a.input:
    for r in csv.DictReader(open(file)):
        key=(tuple(int(r[k]) for k in ['m','n','k']),r['weights'],int(r['bpad']),r['impl'],r['cache'],r['mode'],int(r['repeats']))
        groups[key].append(r)
medians={}
for key,rows in groups.items():
    us=statistics.median(float(r['gpu_us']) for r in rows);m,n,k=key[0]
    medians[key]=dict(gpu_us=us,cpu_us=statistics.median(float(r['cpu_us']) for r in rows),
                     tflops=2*m*n*k/us/1e6,samples=len(rows),
                     private_bytes=int(rows[0]['scratch_bytes']),registers=int(rows[0]['registers']))
selection=json.loads(Path(a.selection).read_text());records=[]
protocols=sorted({(key[-3],key[-2],key[-1]) for key in medians})
for entry in selection:
    shape=tuple(entry[k] for k in ['m','n','k']);fmt=entry['weights'];pad=entry['bpad']
    fixed=entry['fixed_impl'];tuned=entry['tuned_impl']
    for cache,mode,repeats in protocols:
        base=medians[shape,'column',0,fixed,cache,mode,repeats]
        f=medians[shape,fmt,pad,fixed,cache,mode,repeats]
        t=medians[shape,fmt,pad,tuned,cache,mode,repeats]
        record=dict(m=shape[0],n=shape[1],k=shape[2],weights=fmt,bpad=pad,
                    cache=cache,mode=mode,repeats=repeats,fixed_impl=fixed,tuned_impl=tuned,
                    samples=f['samples'],plain_ck_tflops=base['tflops'],
                    fixed_tflops=f['tflops'],tuned_tflops=t['tflops'],
                    fixed_speedup_pct=100*(base['gpu_us']/f['gpu_us']-1),
                    tuned_speedup_pct=100*(base['gpu_us']/t['gpu_us']-1),
                    fixed_private_bytes=f['private_bytes'],tuned_private_bytes=t['private_bytes'])
        affine=[v['tflops'] for key,v in medians.items() if key[0]==shape and key[1]=='column' and key[3].startswith('ck_') and key[-3:]==(cache,mode,repeats)]
        record['best_confirmed_affine_ck_tflops']=max(affine)
        record['tuned_vs_best_affine_pct']=100*(t['tflops']/max(affine)-1)
        vendor=[v['tflops'] for key,v in medians.items() if key[0]==shape and key[3]=='rocblas' and key[-3:]==(cache,mode,repeats)]
        record['best_rocblas_tflops']=max(vendor) if vendor else ''
        records.append(record)
with (out/'comparison.csv').open('w') as file:
    w=csv.DictWriter(file,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
with (out/'configuration-medians.csv').open('w') as file:
    fields=['m','n','k','weights','bpad','impl','cache','mode','repeats','gpu_us','cpu_us','tflops','samples','private_bytes','registers']
    w=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');w.writeheader()
    for (shape,fmt,pad,impl,cache,mode,repeats),v in medians.items():
        w.writerow(dict(m=shape[0],n=shape[1],k=shape[2],weights=fmt,bpad=pad,impl=impl,cache=cache,mode=mode,repeats=repeats,**v))
# Avoid pooling isolated/batched or cold/warm protocols into one conclusion.
for cache,mode,repeats in protocols:
    subset=[r for r in records if r['weights']=='tile-k' and (r['cache'],r['mode'],r['repeats'])==(cache,mode,repeats)]
    fixed=[r['fixed_speedup_pct'] for r in subset];tuned=[r['tuned_speedup_pct'] for r in subset]
    practical=[r['tuned_vs_best_affine_pct'] for r in subset]
    print(cache,mode,repeats,'tile-k versus best confirmed affine min/median/max',tuple(round(x,2) for x in (min(practical),statistics.median(practical),max(practical))))
    print(cache,mode,repeats,'tile-k versus plain CK: fixed min/median/max',
          tuple(round(x,2) for x in (min(fixed),statistics.median(fixed),max(fixed))),
          'tuned min/median/max',tuple(round(x,2) for x in (min(tuned),statistics.median(tuned),max(tuned))))
print(out/'comparison.csv')
