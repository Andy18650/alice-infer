#!/usr/bin/env python3
"""Machine-code inventory and async issue/wait checks; not a numerical oracle."""
# This filename must not shadow stdlib inspect during argparse imports.
from pathlib import Path
import json,re,sys
folder=Path(sys.argv[1]);text=(folder/'probe.isa').read_text();meta=(folder/'metadata.txt').read_text();out=[]
header=(folder/'compute.hpp').read_text()
depth=int(re.search(r'kDepth=(\d+)',header)[1]);stage=int(re.search(r'kStageK=(\d+)',header)[1])
stages=int(re.search(r'kStages=(\d+)',header)[1])
for section in re.split(r'(?=^0[0-9a-f]+ <)',text,flags=re.M):
    name=section.splitlines()[0] if section else ''
    if not any(f'<{x}' in name for x in ('gemm64(','void lds_grid<','copy_check(')):continue
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
    if '<gemm64(' in name:
        waits=[int(x) for l in body for x in re.findall(r'vmcnt\((\d+)\)',l)]
        steady_vm=4*(stages-3)
        assert 4*(min(stages,15)-1) in waits and steady_vm in waits,(name,waits)
        firstwait=next(i for i,l in enumerate(body) if 'vmcnt(' in l)
        firstread=next(i for i,l in enumerate(body) if l.strip().startswith('ds_read_'))
        assert sum(l.strip().startswith('global_load_') for l in body[:firstwait])==4*min(stages,15)
        assert sum(l.strip().startswith('global_load_') for l in body[firstwait:firstread])==(4 if stages==16 else 0)
        assert max(waits)<=62,(name,'VMEM counter range')
        assert not any(l.strip().startswith('ds_write_') for l in lines)
        assert sum('v_mfma_' in l for l in body)==4*stages,(name,'unexpected inner loop')
        bulk=[]
        for i,l in enumerate(body):
            if not l.strip().startswith('s_cbranch_scc1 '):continue
            disp=int(l.strip().split()[1],0);disp=disp-65536 if disp>=32768 else disp
            if disp>=0:continue
            pc=int(re.search(r'// ([0-9a-fA-F]+):',l)[1],16);target=pc+4+4*disp
            seq=[x for x in body[:i+1] if int(re.search(r'// ([0-9a-fA-F]+):',x)[1],16)>=target]
            if sum('v_mfma_' in x for x in seq)==2*stages:bulk=seq
        assert bulk,(name,'missing unrolled bulk backedge')
        assert sum('s_barrier' in x for x in bulk)==stages
        assert all('lgkmcnt(2)' in x for x in bulk if 'lgkmcnt(' in x),('bulk DS drain',name)
        assert sum(x.strip().startswith('ds_read_b64') for x in bulk)==4*stages
        assert sum('s_sub_u32' in x for x in bulk)==1
        assert sum(x.strip().startswith('s_cbranch') for x in bulk)==1
        for phase,i in enumerate(i for i,x in enumerate(bulk) if 'v_mfma_' in x):
            v=4*(phase%2);line=bulk[i]
            assert f'v[{v}:{v+1}], v[{v+2}:{v+3}]' in line,(name,line)
            assert bulk[i+1].strip().startswith(f'ds_read_b64 v[{v}:{v+1}]'),('not immediate same-buffer refill',name)
            assert bulk[i+2].strip().startswith(f'ds_read_b64 v[{v+2}:{v+3}]'),('not immediate same-buffer refill',name)
        assert all(f'vmcnt({steady_vm})' in x for x in bulk if 'vmcnt(' in x)
        assert not any(x.strip().startswith('s_nop') for x in bulk),('bulk explicit M0 gap retained',name)
        copy_indices=[i for i,x in enumerate(bulk) if x.strip().startswith('global_load_')]
        assert len(copy_indices)==4*stages
        for i in copy_indices[::4]:
            assert bulk[i-1].strip().startswith('s_barrier'),('M0 use not directly after barrier',name)
            assert bulk[i-2].strip().startswith('s_add_u32 m0,'),('M0 setup not directly before barrier',name)
        refill_offsets=[int(x.split('//')[0].split(',')[-1].strip(),0) for x in bulk if x.strip().startswith('s_add_u32 m0,')]
        assert refill_offsets==[4096*p for p in range(stages)],('incorrect chunk M0 base',name,refill_offsets)
        load_offsets=[int(re.search(r'offset:(\d+)',bulk[i])[1]) if 'offset:' in bulk[i] else 0 for i in copy_indices]
        assert load_offsets==[q for p in range(stages) for q in (0,256,2048,2304)],('incorrect chunk displacement',name,load_offsets)
        for p in range(stages):
            assert refill_offsets[p]+load_offsets[4*p]==4096*p
            assert refill_offsets[p]+load_offsets[4*p+3]==4096*p+2304
        adds=[x for x in bulk if x.strip().startswith('v_add_co_u32')]
        carries=[x for x in bulk if x.strip().startswith('v_addc_co_u32')]
        assert len(adds)==len(carries)==stages,('source advance not per K16 chunk',name)
        assert all('0x1000' in x or '4096' in x for x in adds)
        offsets=[int(re.search(r'offset:(\d+)',x)[1]) if 'offset:' in x else 0 for x in bulk if x.strip().startswith('ds_read_b64')]
        assert offsets==[4096*((p+1)%stages)+f*2048 for p in range(stages) for f in (0,1) for _ in (0,1)],(name,offsets)
    (folder/('core-'+str(len(out))+'.txt')).write_text('\n'.join(body)+'\n')
    out.append(dict(name=name,direct=direct,copy_check=iscopy,core_lines=len(body),global_to_lds=sum(bool(re.search(r'global_load_.*\blds\s+//',l)) for l in body)))
assert depth==2 and stage==16 and stages in (8,16)
assert len(out)==4,len(out)
resources=[]
for block in re.split(r'(?=  - \.agpr_count:)',meta):
    if not (m:=re.search(r'\.name:\s*(\S+)',block)):continue
    fields={k:int(re.search(r'\.'+k+r':\s*(\d+)',block)[1]) for k in ('vgpr_count','agpr_count','private_segment_fixed_size','group_segment_fixed_size')}
    assert fields['private_segment_fixed_size']==fields['group_segment_fixed_size']==0,(m[1],fields)
    resources.append(dict(name=m[1],**fields))
(folder/'inventory.json').write_text(json.dumps(dict(kernels=out,resources=resources),indent=2)+'\n')
print(f'Inspected {len(out)} direct-only kernels: no scratch/static LDS/lane spills, accumulators stay in AGPRs inside timing; direct kernels have true LDS-target VMEM, K16 two-MFMA/one-barrier chunk checks pass.')
