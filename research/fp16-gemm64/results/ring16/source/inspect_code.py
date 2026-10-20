#!/usr/bin/env python3
"""Machine-code inventory and async issue/wait checks; not a numerical oracle."""
# This filename must not shadow stdlib inspect during argparse imports.
from pathlib import Path
import json,re,sys
folder=Path(sys.argv[1]);text=(folder/'probe.isa').read_text();meta=(folder/'metadata.txt').read_text();out=[]
header=(folder/'compute.hpp').read_text()
depth=int(re.search(r'kDepth=(\d+)',header)[1]);stage=int(re.search(r'kStageK=(\d+)',header)[1])
for section in re.split(r'(?=^0[0-9a-f]+ <)',text,flags=re.M):
    name=section.splitlines()[0] if section else ''
    if not any(f'<{x}' in name for x in ('void gemm<','void gemm_k8<','void lds_grid<','copy_check(')):continue
    lines=[l for l in section.splitlines() if re.search(r'// [0-9A-Fa-f]+:',l)]
    ops=[l.strip().split()[0] for l in lines]
    assert not any(x in ('v_readlane_b32','v_writelane_b32') for x in ops)
    clocks=[i for i,x in enumerate(ops) if x=='s_memtime']
    iscopy='<copy_check(' in name
    if not iscopy:
        assert len(clocks)==2
        body=lines[clocks[0]:clocks[1]]
        assert not any('v_accvgpr_read_b32' in l or 'v_accvgpr_write_b32' in l for l in body),('accumulator spill/copy in timed interval',name)
        assert sum('v_mfma_f32_32x32x8f16' in l for l in body)>0
        for i,line in enumerate(body):
            if 'v_mfma_f32_32x32x8f16' in line:
                if '<void gemm_k8<' in name:
                    assert 's_waitcnt lgkmcnt(0)' in body[i+1],('next prefetch not retired after MFMA',name,line)
                else:
                    assert i>0 and f's_waitcnt lgkmcnt({2*(depth-1)})' in body[i-1],('not direct LDS->MFMA',name,line)
    else:body=lines
    assert not any(l.strip().startswith('v_cvt') for l in lines),('unexpected dtype conversion',name)
    if '<void lds_grid<' in name:assert not any(l.strip().startswith('global_load_') for l in lines)
    direct='<void lds_grid<' not in name
    if direct:
        loads=[i for i,l in enumerate(body) if l.strip().startswith('global_load_')]
        assert loads and all(re.search(r'\blds\s+//',body[i]) for i in loads),name
        assert not any(l.strip().startswith('ds_write_') for l in body),name
        assert any('s_waitcnt vmcnt(0)' in l for l in body)
    if '<void gemm<2>' in name:
        # Future direct copies precede this compute block without a hidden
        # VMEM drain, followed by an explicit VMEM completion wait afterwards.
        first=next(i for i,l in enumerate(body) if 'v_mfma_' in l)
        lastload=max(i for i in loads if i<first)
        assert not any('vmcnt(0)' in l or 's_barrier' in l for l in body[lastload+1:first]),name
        last=max(i for i,l in enumerate(body) if 'v_mfma_' in l)
        assert any('vmcnt(0)' in l for l in body[last+1:]),name
    if m:=re.search(r'<void gemm_k8<(\d+)>',name):
        stages=int(m[1]);waits=[int(x) for l in body for x in re.findall(r'vmcnt\((\d+)\)',l)]
        assert 2*(stages-1) in waits and 2*(stages-2) in waits,(name,waits)
        assert not any(l.strip().startswith('ds_write_') for l in lines)
        assert sum('v_mfma_' in l for l in body)==2*stages,(name,'unexpected inner loop')
        bulk=[]
        for i,l in enumerate(body):
            if not l.strip().startswith('s_cbranch_scc1 '):continue
            disp=int(l.strip().split()[1],0);disp=disp-65536 if disp>=32768 else disp
            if disp>=0:continue
            pc=int(re.search(r'// ([0-9a-fA-F]+):',l)[1],16);target=pc+4+4*disp
            seq=[x for x in body[:i+1] if int(re.search(r'// ([0-9a-fA-F]+):',x)[1],16)>=target]
            if sum('v_mfma_' in x for x in seq)==stages:bulk=seq
        assert bulk,(name,'missing unrolled bulk backedge')
        assert sum('s_barrier' in x for x in bulk)==stages
        assert sum(x.strip().startswith('ds_read_b64') for x in bulk)==2*stages
        assert sum('s_sub_u32' in x for x in bulk)==1
        assert sum(x.strip().startswith('s_cbranch') for x in bulk)==1
        for phase,line in enumerate(x for x in bulk if 'v_mfma_' in x):
            v=4*(phase%2)
            assert f'v[{v}:{v+1}], v[{v+2}:{v+3}]' in line,(name,line)
        assert all(f'vmcnt({2*(stages-2)})' in x for x in bulk if 'vmcnt(' in x)
        offsets=[int(re.search(r'offset:(\d+)',x)[1]) if 'offset:' in x else 0 for x in bulk if x.strip().startswith('ds_read_b64')]
        assert offsets==[2048*((p+1)%stages) for p in range(stages) for _ in (0,1)],(name,offsets)
    (folder/('core-'+str(len(out))+'.txt')).write_text('\n'.join(body)+'\n')
    out.append(dict(name=name,direct=direct,copy_check=iscopy,core_lines=len(body),global_to_lds=sum(bool(re.search(r'global_load_.*\blds\s+//',l)) for l in body)))
expected=3+len(re.findall(r'if constexpr\(Stages==\d+\)',header)) if stage==8 else 5
assert len(out)==expected,(len(out),expected)
resources=[]
for block in re.split(r'(?=  - \.agpr_count:)',meta):
    if not (m:=re.search(r'\.name:\s*(\S+)',block)):continue
    fields={k:int(re.search(r'\.'+k+r':\s*(\d+)',block)[1]) for k in ('vgpr_count','agpr_count','private_segment_fixed_size','group_segment_fixed_size')}
    assert fields['private_segment_fixed_size']==fields['group_segment_fixed_size']==0,(m[1],fields)
    resources.append(dict(name=m[1],**fields))
(folder/'inventory.json').write_text(json.dumps(dict(kernels=out,resources=resources),indent=2)+'\n')
print(f'Inspected {len(out)} direct-only kernels: no scratch/static LDS/lane spills, accumulators stay in AGPRs inside timing; direct kernels have true LDS-target VMEM, and double-buffer future issue has no pre-compute VMEM drain.')
