#!/usr/bin/env python3
"""Validate producer batching and record publication/atomic code in fused kernels."""
import csv
from pathlib import Path
import re
import sys

out=Path(sys.argv[1])
functions=re.split(r'(?=^0[0-9a-f]+ <)',(out/'disassembly.txt').read_text(),flags=re.M)
with (out/'fused-codegen.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['format','batch','weight_loads_before_wait','widths','atomic_adds','cta_barriers','vector_cache_invalidations','scalar_cache_invalidations','lane_reads','lane_writes','acc_reads','acc_writes'])
    for fmt in range(3):
        for batch in [4,8]:
            signature=f'last_arriver<(gemv_layout::Format){fmt}, {batch}>'
            code=next(c for c in functions if signature in c)
            (out/f'fused-format-{fmt}-batch-{batch}.txt').write_text(code)
            if re.search(r'\bscratch_',code):raise RuntimeError('scratch in '+signature)
            first=code.index('\tglobal_load_');wait=code.index('\ts_waitcnt vmcnt',first)
            loads=re.findall(r'global_load_(dword(?:x[24])?)',code[first:wait])
            widths={'dword':4,'dwordx2':8,'dwordx4':16}
            if sum(widths[x] for x in loads)!=16*batch:raise RuntimeError('unexpected producer batch '+signature)
            atomic=len(re.findall(r'\b(?:global|flat)_atomic_add\b',code))
            barriers=len(re.findall(r'\bs_barrier\b',code))
            if not atomic or not barriers:raise RuntimeError('missing atomic/barrier in '+signature)
            writer.writerow([fmt,batch,len(loads),' '.join(str(widths[x]) for x in loads),atomic,barriers,
                             code.count('buffer_wbinvl1'),code.count('s_dcache_inv'),code.count('v_readlane_b32'),code.count('v_writelane_b32'),
                             code.count('v_accvgpr_read_b32'),code.count('v_accvgpr_write_b32')])
            print(signature,'weight loads',len(loads),'atomic adds',atomic,'CTA barriers',barriers)
