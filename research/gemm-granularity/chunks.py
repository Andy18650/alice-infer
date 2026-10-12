#!/usr/bin/env python3
"""Matched parent work; M/N shards are independent, K shards include combine."""
import argparse
from pathlib import Path
import subprocess
import csv
p=argparse.ArgumentParser()
p.add_argument('out');p.add_argument('--rounds',type=int,default=5)
a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
run=Path(__file__).with_name('run.sh').resolve()
# Include a moderate parent and a near-dense-plateau parent.
for parent in [(512,4096,4096),(2048,8192,4096)]:
    for axis in ['m','n','k']:
        for q in [1,2,4,8]:
            dims=list(parent);dims['mnk'.index(axis)]//=q
            for streams in sorted({1,min(q,4)}):
                name=f'{parent[0]}x{parent[1]}x{parent[2]}-{axis}-q{q}-s{streams}'
                subprocess.run(['bash',str(run),str(out/name),'chunks',
                    '--m',str(dims[0]),'--n',str(dims[1]),'--k',str(dims[2]),
                    '--axis',axis,'--q',str(q),'--streams',str(streams),
                    '--rounds',str(a.rounds),'--repeats','16'],check=True)
with (out/'results.csv').open('w') as dest:
    writer=None
    for file in sorted(out.glob('*/results.csv')):
        reader=csv.DictReader(file.open())
        if writer is None:
            writer=csv.DictWriter(dest,fieldnames=reader.fieldnames,lineterminator='\n');writer.writeheader()
        writer.writerows(reader)
print(out/'results.csv')
