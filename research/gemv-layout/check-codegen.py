#!/usr/bin/env python3
"""Inspect actual weight batches; x loads in GEMV may also use VMEM."""
import csv
from pathlib import Path
import re
import sys

out=Path(sys.argv[1])
code=(out/'disassembly.txt').read_text()
functions=re.split(r'(?=^0[0-9a-f]+ <)',code,flags=re.M)
with (out/'codegen.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['format','batch','mode','loads_before_first_wait','widths','first_wait','segment_global_loads','segment_waits','lane_reads','lane_writes','acc_reads','acc_writes','cache_flags','fp32_fma_instructions','scalar_load_instructions','packed_fp32_fma_instructions'])
    for fmt in range(3):
        for batch in [4,8]:
            for mode in range(3):
                signature=f'partial<(gemv_layout::Format){fmt}, {batch}, (gemv_layout::Mode){mode}>'
                text=next(c for c in functions if signature in c)
                if re.search(r'\bscratch_',text):raise RuntimeError('scratch: '+signature)
                first=text.index('\tglobal_load_')
                wait=text.index('\ts_waitcnt vmcnt',first)
                prefix=text[first:wait]
                loads=re.findall(r'global_load_(dword(?:x[24])?)',prefix)
                widths={'dword':4,'dwordx2':8,'dwordx4':16}
                # In this implementation uniform FP32 x uses SMEM, so the
                # VMEM prefix consists only of weights in all three modes.
                if sum(widths[x] for x in loads)!=16*batch:
                    raise RuntimeError('unexpected/serialized weight batch: '+signature+' '+str(loads))
                lines=[]
                for line in text[first:].splitlines():
                    lines.append(line)
                    branch=re.search(r'\bs_(?:cbranch_[a-z0-9]+|branch)\s+(-?(?:0x[0-9a-f]+|[0-9]+))',line)
                    if branch and (int(branch[1],0)<0 or int(branch[1],0)>32767):break
                segment='\n'.join(lines)
                flags=sorted(set(re.findall(r'\b(?:glc|slc|dlc)\b',segment)))
                fmas=len(re.findall(r'\bv_(?:pk_)?(?:fma|fmac)_f32',text))
                if mode==2 and not fmas:raise RuntimeError('missing FP32 FMA: '+signature)
                writer.writerow([fmt,batch,mode,len(loads),' '.join(str(widths[x]) for x in loads),re.search(r'vmcnt\((\d+)\)',text[wait:])[1],
                                 len(re.findall('global_load_',segment)),' '.join(re.findall(r'vmcnt\((\d+)\)',segment)),
                                 text.count('v_readlane_b32'),text.count('v_writelane_b32'),text.count('v_accvgpr_read_b32'),text.count('v_accvgpr_write_b32'),' '.join(flags) or 'none',fmas,len(re.findall(r'\bs_load_',text)),text.count('v_pk_fma_f32')])
                print(signature,'loads before first wait',len(loads),'widths',sorted({widths[x] for x in loads}))
