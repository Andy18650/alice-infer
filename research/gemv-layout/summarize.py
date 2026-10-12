#!/usr/bin/env python3
import collections
import csv
from pathlib import Path
import statistics
import sys

out=Path(sys.argv[1])
keys=['shape','N','K','format','batch','splits','grid','jobs','padded_bytes','useful_bytes','phase','iters']
groups=collections.defaultdict(list)
for row in csv.DictReader((out/'raw.csv').open()):
    groups[tuple(row[k] for k in keys)].append(row)
with (out/'summary.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(keys+['samples','median_ms','median_wall_ms','median_GB_s','median_packed_GB_s','max_abs_error'])
    for key,rows in sorted(groups.items()):
        med=lambda field:statistics.median(float(r[field]) for r in rows)
        writer.writerow(list(key)+[len(rows),med('ms'),med('wall_ms'),med('useful_GB_s'),med('packed_GB_s'),max(float(r['max_abs_error']) for r in rows)])
    for shape in sorted({k[0] for k in groups}):
        print(shape)
        for phase in ['read','expand','gemv_partial','gemv_full','gemv_last','gemv_last_spaced']:
            matching=[(k,v) for k,v in groups.items() if k[0]==shape and k[-2]==phase]
            if not matching:continue  # Historical reports predate fused phases.
            k,rows=max(matching,key=lambda p:statistics.median(float(r['useful_GB_s']) for r in p[1]))
            print(' ',phase,'format/batch/splits/grid',k[3:7],f"{statistics.median(float(r['useful_GB_s']) for r in rows):.1f} GB/s")
