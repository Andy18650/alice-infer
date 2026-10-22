#!/usr/bin/env python3
"""Summarize measured kernel rates and observed CTA-body overlap."""
import argparse,csv,hashlib,json,struct
from pathlib import Path
here=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='?',default=here/'results/final');a=p.parse_args();out=a.results.resolve()
rows=list(csv.DictReader((out/'summary.csv').open()));timing=[x for x in rows if x['kind']=='timing'];text=['# Native GEMM matched results','',
'Native HIP O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, no model/context/clock overrides. CPU packing/H2D/reference excluded; full kernel HIP event used for TFLOP/s. Two counterbalanced process repeats, nine samples each.','',
'| M x N x K | Tile | Useful TFLOP/s run1/run2 | Effective core MHz | Global clock-adjusted efficiency |',
'|---|---:|---:|---:|---:|']
keys=sorted({(int(x['m']),int(x['n']),int(x['k'])) for x in timing})
for m,n,k in keys:
 for tile in (128,256):
  r=sorted([x for x in timing if (int(x['m']),int(x['n']),int(x['k']),int(x['tile']))==(m,n,k,tile)],key=lambda x:int(x['repeat']))
  if r:text.append(f'|{m}x{n}x{k}|{tile}|'+ ' | '.join('/'.join(f'{float(x[key])*scale:.2f}' for x in r) for key,scale in (('useful_TFLOPs',1),('median_core_MHz',1),('global_clock_adjusted_efficiency',100)))+'%|')
proof=[]
for tile in (128,256):
 file=out/'clocks'/f't{tile}-m4096-n6656-k32768-r1-a1729-b2718-r4.bin';b=file.read_bytes();t=struct.unpack('<'+'Q'*(len(b)//8),b);events=[]
 for i in range(len(t)//16):
  v=t[i*16:(i+1)*16];start=min(v[w*4+2] for w in range(4));end=max(v[w*4+3] for w in range(4));events.extend(((start,1),(end,-1)))
 live=peak=0
 for _,d in sorted(events):live+=d;peak=max(peak,live)
 proof.append(dict(tile=tile,file=str(file),sha256=hashlib.sha256(b).hexdigest(),peak_overlapping_cta_bodies=peak,ctas=len(t)//16))
text+=['','Observed maximum simultaneous CTA-body intervals in the representative long-K launch: '+', '.join(f"tile{p['tile']}={p['peak_overlapping_cta_bodies']}" for p in proof)+'. This is body-overlap evidence, not exact allocation timestamps or a CTA-to-CU map.',
'',f"Checked launches: {sum(int(x['samples']) for x in rows)}. Large cases use sampled exact references plus all-C finite/poison and guard scans; do not relabel them full references."]
(out/'overlap.json').write_text(json.dumps(proof,indent=2)+'\n');(out/'summary.md').write_text('\n'.join(text)+'\n');print('\n'.join(text))
