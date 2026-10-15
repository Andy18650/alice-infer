#!/usr/bin/env python3
"""Audit the measured spill-free V3 256x256x64 AB-panel hot loop.

Count an actual K-step backedge, not a larger backward control-flow edge that
happens to include prologue/tail/epilogue code. Static counts are not stall
cycles or performance-counter measurements.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
p=argparse.ArgumentParser();p.add_argument('object');p.add_argument('out');a=p.parse_args()
obj=Path(a.object).resolve();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
dump=shutil.which('llvm-objdump-21') or shutil.which('llvm-objdump')
read=shutil.which('llvm-readobj-21') or shutil.which('llvm-readobj')
text=subprocess.check_output([dump,'-d','--demangle',str(obj)],text=True)
meta=subprocess.check_output([read,'--notes',str(obj)],text=True)
(out/'disassembly.txt').write_text(text);(out/'metadata.txt').write_text(meta)
section=next(s for s in re.split(r'(?=^0[0-9a-f]+ <)',text,flags=re.M)
             if 'entry<(compute_bound::Storage)2>' in s[:500])
ins=[]
for line in section.splitlines():
    m=re.search(r'// ([0-9a-fA-F]+):',line)
    if m:ins.append((int(m[1],16),line))
expected=256*256*64//(4*32*32*8)
loops=[]
for pc,line in ins:
    m=re.search(r'\bs_(?:cbranch_\w+|branch)\s+(-?\d+)',line)
    if not m:continue
    imm=int(m[1]);imm=imm-65536 if imm>32767 else imm
    target=pc+4+imm*4
    if target>=pc:continue
    body=[l for addr,l in ins if target<=addr<=pc]
    n=sum('v_mfma_f32_32x32x8bf16_1k' in l for l in body)
    if n==expected:loops.append((len(body),target,pc,body))
if not loops:raise SystemExit('No backedge with the expected K-step MFMA count')
_,start,end,body=min(loops)
counts=collections.Counter(re.search(r'^\s*(\S+)',l)[1] for l in body)
waits=collections.Counter(re.findall(r'(?:vmcnt|lgkmcnt)\(\d+\)','\n'.join(body)))
if counts['buffer_load_dwordx4']!=16 or counts['ds_read_b128']!=32 or counts['ds_write_b128']!=16:
    raise SystemExit('Unexpected working-kernel load/staging counts')
# Metadata is authoritative for scratch, unlike mnemonic-only searches on gfx90a.
# Parse individual metadata kernel blocks instead of consuming earlier entries.
blocks=re.split(r'(?=  - \.agpr_count:)',meta)
block=next(b for b in blocks if re.search(r'\.name:\s*\S*StorageE2\S*',b))
fields={name:int(re.search(r'\.'+name+r':\s*(\d+)',block)[1]) for name in
        ['agpr_count','group_segment_fixed_size','private_segment_fixed_size','sgpr_count','vgpr_count','wavefront_size']}
if fields['private_segment_fixed_size']!=0:raise SystemExit('Working AB-panel entry spills')
summary={'object':str(obj),'sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),
         'hot_loop_start':hex(start),'hot_loop_end':hex(end),'static_instructions':len(body),
         'instruction_counts':dict(counts),'wait_thresholds':dict(waits),'metadata':fields,
         'flops_per_cta_k_step':2*256*256*64,
         'global_requested_bytes_per_cta_k_step':16*16*256,
         'lds_read_bytes_per_cta_k_step':32*16*256,
         'lds_write_bytes_per_cta_k_step':16*16*256,
         'warning':'Requested bytes/static issue order are not physical traffic or stall-cycle counters.'}
(out/'hot-loop.txt').write_text('\n'.join(body)+'\n')
(out/'memory-math-sequence.txt').write_text('\n'.join(l for l in body if re.search(r'buffer_load_|ds_(read|write)|v_mfma_|s_waitcnt|s_barrier',l))+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
