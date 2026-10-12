#!/usr/bin/env python3
"""Inspect extracted code objects, not source intent or the host executable."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
p=argparse.ArgumentParser()
p.add_argument('root');p.add_argument('--candidates',type=int,default=7)
p.add_argument('--split-entries',action='store_true')
p.add_argument('--packed-entries',action='store_true')
p.add_argument('--allow-packed-private-bytes',type=int,default=0)
a=p.parse_args();root=Path(a.root);ck_count=0;split_count=0;packed_count=0;reference_count=0
tool=shutil.which('llvm-readobj-21') or shutil.which('llvm-readobj')
if not tool:raise SystemExit('llvm-readobj is required for private-segment checks')
for file in sorted(root.glob('*.disassembly.txt')):
    obj=Path(str(file).removesuffix('.disassembly.txt'))
    metadata=subprocess.check_output([tool,'--notes',str(obj)],text=True)
    private={}
    for name,size in re.findall(r'\.name:\s*(\S+)\s+\.private_segment_fixed_size:\s*(\d+)',metadata):
        if 'mfma_reference' in name:kind='reference'
        elif 'packed_entry' in name:kind='packed'
        elif 'split_entry' in name:kind='split'
        elif 'GLOBAL__N_15entry' in name:kind='regular'
        else:continue
        private[kind]=int(size)
    for function in re.split(r'(?=^0[0-9a-f]+ <)',file.read_text(),flags=re.M):
        title=function[:500]
        if 'mfma_reference(float*, int)' in title:
            n=len(re.findall(r'\bv_mfma_f32_16x16x16bf16_1k\b',function))
            if n!=4:raise SystemExit(f'Expected exactly 4 distinct reference-chain MFMA instructions, got {n}')
            if private.get('reference',-1)!=0 or re.search(r'\bscratch_',function):raise SystemExit('Reference has a private segment or scratch instruction')
            reference_count+=1;print(file.name,'reference MFMA count',n)
        elif any(s in title for s in ['granularity::(anonymous namespace)::entry', 'granularity::(anonymous namespace)::split_entry', 'granularity::(anonymous namespace)::packed_entry']):
            n=len(re.findall(r'\bv_mfma_f32_\w*bf16\w*\b',function))
            if not n:raise SystemExit('CK kernel lacks BF16 MFMA')
            split='::split_entry' in title;packed='::packed_entry' in title
            kind='packed' if packed else 'split' if split else 'regular'
            if kind not in private:raise SystemExit('Missing kernel private-segment metadata')
            size=private[kind];budget=a.allow_packed_private_bytes if packed else 0
            if size>budget:raise SystemExit(f'{kind} kernel private segment {size} exceeds budget {budget}')
            if size:print('WARN',file.name,kind,'private bytes/thread',size)
            if not size and re.search(r'\bscratch_',function):raise SystemExit('Unexpected scratch instruction')
            if packed and a.packed_entries and 'buffer_load_ushort' in function:
                raise SystemExit('Packed CK entry scalarized BF16 loads')
            split_count+=int(split);packed_count+=int(packed);ck_count+=int(not split and not packed)
            print(file.name,'packed' if packed else 'split' if split else 'regular','CK static BF16 MFMA count',n)
if ck_count!=a.candidates or reference_count!=1 or (a.split_entries and split_count!=a.candidates) or (a.packed_entries and packed_count!=a.candidates):
    raise SystemExit(f'Expected {a.candidates} CK entries, 1 reference and optional split/packed twins; got {ck_count}/{reference_count}/{split_count}/{packed_count}')
print('PASS: native BF16 MFMA, four reference chains, private-segment metadata within explicit budgets')
