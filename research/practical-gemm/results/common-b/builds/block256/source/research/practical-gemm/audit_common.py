#!/usr/bin/env python3
"""Bounded B coordinates/pointer/lifetime checks plus actual ISA/resources.

Not a cycle simulator or a peer/memory-order proof. GPU oracle/primitive tests
remain independent gates. No result path is overwritten.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('common_b_generator',HERE/'common_b.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
from inspect_code import parse_instructions,find_hot_loop,opcode_counts


def index(fmt,k,n,sn):
    if fmt=='plain':return k*sn+n
    if fmt.startswith('block'):
        bn=int(fmt[5:]);return ((k//128)*(sn//bn)+n//bn)*128*bn+(k%128)*bn+n%bn
    return ((k//32)*(sn//128)+n//128)*4096+(k%32//8)*1024+(n%32+(k%8//4)*32)*16+(n%128//32)*4+k%4


def coordinates():
    seen={}
    for w in range(4):
        for lane in range(64):
            base=8192+w*2048+(lane&7)*64+(lane&8)+(lane&16)*64+(lane&32)*16
            for col in range(4):
                for h in range(4):
                    n=4*(lane%32)+col;k=w*8+(lane//32)*4+h
                    address=base+col*16+h*2
                    old=8192+(k//8)*2048+(n//64)*1024+(n%32+(k%8//4)*32)*16+(n%64//32)*8+(k%4)*2
                    assert address==old and address not in seen
                    seen[address]=(k,n)
    assert set(seen)==set(range(8192,16384,2))
    count=0
    for fmt in common.FORMATS:
        bn=256 if fmt=='block256' else 128
        for n in (1,129,257,769):
            sn=(n+bn-1)//bn*bn
            for k in (32,160,352):
                sk=(k+127)//128*128 if fmt.startswith('block') else k
                visited=bytearray(sn*sk)
                for kk in range(sk):
                    for nn in range(sn):
                        p=index(fmt,kk,nn,sn)
                        assert 0<=p<len(visited) and visited[p]==0
                        visited[p]=1;count+=1
                assert all(visited)
                # Independently validate actual per-wave four-DWORD addressing.
                if fmt!='native':
                    stride=sn if fmt=='plain' else bn
                    for stage in range(k//32):
                        for tile in range((n+127)//128):
                            panel=tile*128 if fmt=='plain' else (tile//(bn//128))*128*bn+(tile%(bn//128))*128
                            for w in range(4):
                                head=stage*32*sn if fmt=='plain' else (stage//4)*128*sn+(stage%4)*32*bn
                                for lane in range(64):
                                    off=((lane//32)*(16*stride-512)+lane*16)//4
                                    for j in range(4):
                                        actual=panel+head+w*8*stride+off+j*stride
                                        assert actual==index(fmt,stage*32+w*8+(lane//32)*4+j,tile*128+(lane%32)*4,sn)
    return count


def trace(code,stages,full):
    labels={op[:-1]:i for i,op in enumerate(code) if op.endswith(':')}
    value={'s16':stages,'%[tiles]':stages,'%[a_full]':int(full),'%[rows_left]':64,'%[cols_left]':64}
    pc=0;scc=False;executed=[]
    while pc<len(code):
        op=code[pc];pc+=1
        if op.endswith(':'):continue
        if op.startswith('s_cmp_'):
            compare,args=op.split(' ',1);a,b=args.split(', ')
            av=value[a] if a in value else int(a)
            bv=value[b] if b in value else int(b)
            scc={'s_cmp_eq_u32':av==bv,'s_cmp_ge_u32':av>=bv,'s_cmp_le_u32':av<=bv}[compare]
        elif op=='s_sub_u32 s16, s16, 8':value['s16']-=8
        elif op=='s_sub_u32 s16, s16, 1':value['s16']-=1
        elif op.startswith(('s_branch ','s_cbranch_scc0 ','s_cbranch_scc1 ')):
            branch,target=op.split(' ',1)
            if branch=='s_branch' or scc==(branch=='s_cbranch_scc1'):pc=labels[target]
        executed.append(op)
        assert len(executed)<stages*400+10000,'unbounded trace'
    return executed


def projection(op):
    return op.startswith(('v_mfma_','v_pk_fma_f32','ds_read_b128','s_waitcnt','s_barrier',
                          's_load_dword s14','s_mov_b32 s15,','s_add_u32 s12,','s_addc_u32 s13,'))


def staging(ops,fmt,stages,sn):
    bn=256 if fmt=='block256' else 128
    stride=sn if fmt=='plain' else bn
    ptr=0;loads=0;producer=0;pending=set();owner=None;consumed=True;copied=set();progress={};stores=set()
    for op in ops:
        if op.startswith(('global_load_dword v6','global_load_dwordx4 v[60:63]')):
            register=60 if 'x4 ' in op else int(op.split('v')[1].split(',')[0])
            if register==60:
                assert consumed and not pending,'B reload before current raw copies complete'
                expected=loads*32*sn if fmt in ('native','plain') else (loads//4)*128*sn+(loads%4)*32*bn
                assert ptr==expected,('wrong B stage pointer',fmt,loads,ptr,expected)
                assert loads<stages,'B overfetch'
                loads+=1
            pending.update(range(60,64) if 'x4 ' in op else [register])
        elif op=='s_waitcnt vmcnt(0)' and pending:
            assert pending==set(range(60,64));pending.clear();owner=loads-1;consumed=False;copied.clear()
        elif op.startswith('s_add_u32 s2,'):
            ptr+=128*sn-96*bn if 'b_group_step' in op else 32*(sn if fmt in ('native','plain') else bn)
        elif op.startswith('v_perm_b32') and ', 0, ' in op:
            assert not pending and owner==producer,'B expansion from pending/wrong stage'
            dst=int(op.split()[1].rstrip(',')[1:]);copied.add(dst);progress[dst]=0
            if len(copied)==8:consumed=True
        elif op.startswith(('v_pk_ashrrev_i16','v_pk_mul_f16')) or op.startswith('v_and_b32') and '%[mask]' in op:
            dst=int(op.split()[1].rstrip(',')[1:]);progress[dst]+=1
        elif op.startswith(('ds_write_b128','ds_write_b64')) and 'offset:8' in op or op.startswith('ds_write_b128') and 'offset:9216' in op:
            # B stores only; A offsets are0/256/512/768. Native second B store
            # uses9216, not the common producer's8208/8224/8240.
            reg=int(op.split('v[')[1].split(':')[0]);width=4 if 'b128' in op else 2
            assert all(progress[r]==3 for r in range(reg,reg+width)),'B store before conversion'
            stores.update(range(reg,reg+width))
            if stores==set(range(52,60)):stores.clear();producer+=1
    assert not pending and producer==stages and loads==stages


def resources(text):
    result={}
    for block in text.split('  - .agpr_count:')[1:]:
        name=re.search(r'\.name:\s*(\S+)',block)[1]
        if not any(s in name for s in ('native_gemm','decode_partial','decode_combine')):continue
        item={k:int(re.search(r'\.'+k+r':\s*(\d+)',block)[1]) for k in
              ('vgpr_count','sgpr_count','private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count')}
        assert item['private_segment_fixed_size']==item['sgpr_spill_count']==item['vgpr_spill_count']==0
        if 'native_gemm' in name:assert item['vgpr_count']==256
        result[name]=item
    assert len(result)==3
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    generated={};native={};result={'coordinate_cases':coordinates(),'builds':{}}
    for fmt in common.FORMATS:
        emitter=common.practical.RowMajorEmitter() if fmt=='native' else common.CommonBEmitter(fmt)
        text=emitter.render();generated[fmt]=emitter.instructions
        directory=a.build_root/fmt
        assert text==(directory/'compute.hpp').read_text()
        assert common.kernel_text(fmt)==(directory/'kernel.hpp').read_text()
        inst=parse_instructions((directory/'probe.isa').read_text());loop=find_hot_loop(inst,128);native[fmt]=loop
        counts=opcode_counts(loop)
        for op,count in [('v_mfma_f32_32x32x8f16',128),('v_pk_fma_f32',64),('ds_read_b128',64),('s_waitcnt',24),('s_barrier',16)]:
            assert counts[op]==count,(fmt,op,counts[op])
        assert counts['v_perm_b32']==(64 if fmt=='native' else 128)
        assert counts['ds_write_b64']==(32 if fmt=='native' else 64)
        assert not any(re.search(r'accvgpr|scratch_|writelane|readlane',op) for _,op in inst)
        result['builds'][fmt]={'resources':resources((directory/'metadata.txt').read_text()),'static_hot_loop':dict(counts)}
        (a.output/f'{fmt}-loop.txt').write_text('\n'.join(loop)+'\n')
    def native_projection(ops):
        # Caller binds the identical scale_step argument to a different SGPR.
        return [re.sub(r's\d+$','<scale_step>',op) if op.startswith('s_add_u32 s12,') else op
                for op in ops if projection(op)]
    for fmt in common.FORMATS:
        assert native_projection(native[fmt])==native_projection(native['native'])
    counts=0
    for stages in [*range(30),63,64,65,2048]:
        for full in (False,True):
            traces={fmt:trace(code,stages,full) for fmt,code in generated.items()}
            for fmt,ops in traces.items():
                assert [op for op in ops if projection(op)]==[op for op in traces['native'] if projection(op)]
                for sn in (256,768):staging(ops,fmt,stages,sn)
                counts+=1
    result.update(stage_control_cases=counts,generated_native_compute_read_fence_scale_projections_equal=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
