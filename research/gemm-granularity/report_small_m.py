#!/usr/bin/env python3
"""Small-M parent-group medians; useful weight bandwidth is not HBM counters."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics
p=argparse.ArgumentParser()
p.add_argument('--input',nargs='+',required=True)
p.add_argument('--calibration',required=True)
p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
log=Path(a.calibration).read_text()
bw=statistics.median(float(v) for v in re.findall(r'READ_REFERENCE.*?useful_GBps=([\d.]+)',log))
keys=['m','n','k','axis','q','streams','bpad','weights','impl','k_launch','cache','mode','repeats']
groups=collections.defaultdict(list)
for file in a.input:
    for r in csv.DictReader(open(file)):
        r.setdefault('weights','column');groups[tuple(r[k] for k in keys)].append(r)
summary=[]
for key,rows in groups.items():
    r=rows[0];m,n,k,q=map(int,[r[x] for x in ['m','n','k','q']]);axis=r['axis']
    pn=n*(q if axis=='n' else 1);pk=k*(q if axis=='k' else 1)
    us=statistics.median(float(v['gpu_us']) for v in rows)
    weight_bytes=2*pn*pk;modeled_bytes=float(r['modeled_bytes'])
    record={'parent_m':m,'parent_n':pn,'parent_k':pk}
    record.update(dict(zip(keys,key)))
    record.update(samples=len(rows),gpu_us=us,cpu_us=statistics.median(float(v['cpu_us']) for v in rows),
                  weight_GBps=weight_bytes/us/1000,modeled_useful_GBps=modeled_bytes/us/1000,
                  read_reference_fraction=weight_bytes/us/1000/bw,
                  tflops=2*m*pn*pk/us/1e6)
    for field in ['ctas','producer_launches','registers','lds_bytes','scratch_bytes','blocks_per_cu','tile_flops']:
        record[field]=r[field]
    summary.append(record)
with (out/'summary.csv').open('w') as file:
    w=csv.DictWriter(file,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
best={}
for r in summary:
    key=tuple(r[k] for k in ['parent_m','parent_n','parent_k','cache','mode','repeats'])
    if key not in best or r['gpu_us']<best[key]['gpu_us']:best[key]=r
for name,records in [('best.csv',best.values()),('best-ck.csv',[])]:
    if name=='best-ck.csv':
        ck={}
        for r in summary:
            if not r['impl'].startswith('ck_'):continue
            key=tuple(r[k] for k in ['parent_m','parent_n','parent_k','cache','mode','repeats'])
            if key not in ck or r['gpu_us']<ck[key]['gpu_us']:ck[key]=r
        records=ck.values()
    with (out/name).open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(records)
(out/'reference.json').write_text(json.dumps({'read_useful_GBps':bw,
    'warning':'Weight numerator excludes activations, partials, output and padding; time includes full group and combine. Not measured physical HBM bandwidth.'},indent=2)+'\n')
print(f'{len(summary)} configuration medians; read reference {bw:.3f} GB/s')
