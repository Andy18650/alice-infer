#!/usr/bin/env python3
"""Bounded coordinate, generated-isolation and embedded-ISA/resource checks."""
import argparse
import json
import re
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'native-gemm'))
from inspect_code import parse_instructions,find_hot_loop,opcode_counts,integer_field
from generate_q0_schedule import Q0Emitter
# 'generate' is also a native module name: import the study generator by path.
import importlib.util
spec=importlib.util.spec_from_file_location('practical_generator',HERE/'generate.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def coordinate_check():
    seen={}
    for wave in range(4):
        for lane in range(64):
            for packet in range(2):
                row=wave*32+lane//4+packet*16
                q=lane%4
                base=q*2048+(wave//2)*1024+(wave%2)*8+(lane//4)*16+packet*256
                for half in range(8):
                    address=base+(half//4)*512+(half%4)*2
                    assert address not in seen
                    seen[address]=(row,q*8+half)
    assert set(seen)==set(range(0,8192,2))
    for address,(row,k) in seen.items():
        old=(k//8)*2048+(row//64)*1024+(row%32+(k%8//4)*32)*16+(row%64//32)*8+(k%4)*2
        assert address==old
    # Verify the actual MAD/byte-limit formulas for every M-tail and several
    # strides, including zero active rows and maximum supported K.
    checked=0
    for m in range(1,258):
        for k in (32,64,96,352,4096,14336,65536):
            for tile in range((m+127)//128):
                for wave in range(4):
                    rows=max(0,min(32,m-tile*128-wave*32))
                    for lane in range(64):
                        offset=(lane//4)*(k*2-64)+lane*16
                        assert offset==(lane//4)*(k*2)+(lane%4)*16
                        for packet in range(2):
                            active=offset<max(rows-packet*16,0)*k*2
                            row=tile*128+wave*32+lane//4+packet*16
                            assert active==(row<m)
                            if active:
                                address=(tile*128+wave*32)*k*2+offset+packet*16*k*2
                                assert address==row*k*2+(lane%4)*16
                                assert address+16<=m*k*2
                            checked+=1
    return checked


def invariant(op):
    return (op.startswith(('v_mfma_','v_pk_fma_f32','ds_read_b128','s_waitcnt','s_barrier',
                           's_load_dword s14','global_load_dwordx4 v[60:63]'))
            or op.startswith('ds_write_b128') and ('offset:8192' in op or 'offset:9216' in op))


def run(builds,output):
    output.mkdir(parents=True,exist_ok=False)
    base=Q0Emitter('balanced');base.render()
    row=study.RowMajorEmitter();row.render()
    assert [op for op in base.instructions if invariant(op)]==[op for op in row.instructions if invariant(op)]
    results={'coordinate_cases':coordinate_check(),'generated_non_A_projection_equal':True,'builds':{}}
    for layout,directory in builds.items():
        source=(directory/'compute.hpp').read_text()
        # Fresh emitters: render mutates their instruction list.
        expected=(study.RowMajorEmitter() if layout=='rowmajor' else Q0Emitter('balanced')).render()
        assert source==expected,'Generated build header does not match current source'
        instructions=parse_instructions((directory/'probe.isa').read_text())
        loop=find_hot_loop(instructions,128)
        counts=opcode_counts(loop)
        for op,count in [('v_mfma_f32_32x32x8f16',128),('v_pk_fma_f32',64),
                         ('s_barrier',16),('ds_read_b128',64),('s_waitcnt',24),('s_load_dword',2)]:
            assert counts[op]==count,(layout,op,counts[op])
        assert not any(re.search(r'accvgpr|scratch_|writelane|readlane',op) for _,op in instructions)
        if layout=='rowmajor':
            assert counts['ds_write_b64']==32 and counts['ds_write_b128']==16
            assert counts['global_load_dwordx4']==40 # Includes both branches, only24 execute on full rows.
        else:
            assert counts['ds_write_b64']==0 and counts['ds_write_b128']==32
            assert counts['global_load_dwordx4']==24
        notes='  - .agpr_count:'+(directory/'metadata.txt').read_text().split('  - .agpr_count:')[1]
        fields={key:integer_field(notes,r'\.'+key+r':\s*(\d+)') for key in
                ('agpr_count','vgpr_count','sgpr_count','private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count')}
        assert fields['vgpr_count']==256 and fields['agpr_count']==0 and fields['private_segment_fixed_size']==0
        assert fields['sgpr_spill_count']==fields['vgpr_spill_count']==0
        results['builds'][layout]={'resources':fields,'static_hot_loop':dict(counts)}
        (output/f'{layout}-loop.txt').write_text('\n'.join(loop)+'\n')
    packed=parse_instructions((builds['packed']/'probe.isa').read_text())
    row_native=parse_instructions((builds['rowmajor']/'probe.isa').read_text())
    # Compiler binds external SGPRs differently; fixed MFMA/scale/raw operands,
    # waits and B payload remain identical in the actual eight-stage body.
    a=[op for op in find_hot_loop(packed,128) if invariant(op)]
    b=[op for op in find_hot_loop(row_native,128) if invariant(op)]
    assert a==b,'Changed native non-A projection'
    results['native_non_A_projection_equal']=True
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed',type=Path,required=True);p.add_argument('--rowmajor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run({'packed':a.packed,'rowmajor':a.rowmajor},a.output)
