#!/usr/bin/env python3
"""Sequential randomized timing-only matrix, with per-configuration warm-up."""
import argparse
import itertools
from pathlib import Path
import random
import shlex
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=2)
p.add_argument('--rounds',type=int,default=9)
p.add_argument('--steps',type=int,default=4096)
p.add_argument('--lds',action='store_true',help='sweep LDS hot/sweep instead of global modes')
p.add_argument('--warmup-ms',type=int,default=200)
p.add_argument('--verify',type=int,choices=(0,1),default=1)
p.add_argument('--pattern',choices=('legacy','hash'),default='legacy')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
for name in ('hashes.txt','compiler-flags.txt','inventory.json'):
    shutil.copy2(a.binary.parent/name,a.output/name)
with (a.output/'amd-smi-before.txt').open('w') as f:
    subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (a.output/'commands.txt').open('w') as commands:
    for repeat in range(1,a.repeats+1):
        modes=('lds-hot','lds-sweep') if a.lds else ('hot','l2','stream')
        cases=list(itertools.product(modes,(1,2,4)))
        random.Random(1729 if repeat==1 else 2718+repeat-2).shuffle(cases)
        for mode,loads in cases:
            stem=f'{mode}-l{loads}-w1-run{repeat}'
            cmd=['numactl','--cpunodebind=0','--membind=0',str(a.binary.resolve()),
                 '--mode',mode,'--loads',str(loads),'--steps',str(a.steps),
                 '--rounds',str(a.rounds),'--warmup-ms',str(a.warmup_ms),
                 '--verify',str(a.verify),'--pattern',a.pattern]
            commands.write(shlex.join(cmd)+'\n');commands.flush()
            with (a.output/f'{stem}.csv').open('w') as out,(a.output/f'{stem}.log').open('w') as err:
                result=subprocess.run(cmd,stdout=out,stderr=err,check=False)
            if result.returncode:
                raise RuntimeError(f'{stem}: unexpected failure {result.returncode}; see saved log')
            print(stem,flush=True)
with (a.output/'amd-smi-after.txt').open('w') as f:
    subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
