#!/usr/bin/env python3
"""Scalar-control and raw-bank/scale-ownership proof, not a latency emulator."""
import argparse
import collections
import hashlib
import json
import re
import struct
from pathlib import Path
from audit_branch_flow import HERE, ROOT, STAGE_COUNTS, asm_instructions, require


def elf_text(path):
    data=path.read_bytes()
    require(data[:6]==b'\x7fELF\x02\x01','Expected little-endian ELF64')
    offset=struct.unpack_from('<Q',data,40)[0]
    size,count,names=struct.unpack_from('<HHH',data,58)
    def section(index): return struct.unpack_from('<IIQQQQIIQQ',data,offset+index*size)
    strings=section(names)
    table=data[strings[4]:strings[4]+strings[5]]
    for index in range(count):
        item=section(index)
        if table[item[0]:].split(b'\0',1)[0]==b'.text':
            return data[item[4]:item[4]+item[5]]
    raise RuntimeError('Missing .text')


def execute(code,stages):
    labels={op[:-1]:i for i,op in enumerate(code) if op.endswith(':')}
    remaining=stages
    scale=None
    previous=None
    scale_loads=0
    raw={i:(None,0) for i in range(128,256)}
    output={i:[] for i in range(64,128)}
    last_write={}
    mfmas=0
    counts=collections.Counter()
    minimum_distance=None
    pc=0
    scc=False
    steps=0
    while pc<len(code):
        steps+=1
        require(steps<2000000,'Nonterminating control trace')
        op=code[pc];pc+=1
        if op.endswith(':'): continue
        if op=='s_memtime s[6:7]': break
        opcode=op.split()[0]
        if opcode.startswith('s_cmp_'):
            require('s16,' in op,'Unexpected comparison before end clock')
            rhs=int(op.rsplit(',',1)[1])
            scc={'s_cmp_eq_u32':remaining==rhs,'s_cmp_le_u32':remaining<=rhs,
                 's_cmp_ge_u32':remaining>=rhs}[opcode]
        elif opcode=='s_cbranch_scc1':
            if scc: pc=labels[op.split()[1]]
        elif opcode=='s_branch': pc=labels[op.split()[1]]
        elif op.startswith('s_sub_u32 s16'):
            remaining-=int(op.rsplit(',',1)[1]);require(remaining>=0,'Counter underflow')
        elif op=='s_mov_b32 s15, s14': previous=scale
        elif op=='s_load_dword s14, s[12:13], 0':
            scale=scale_loads;scale_loads+=1
        elif opcode.startswith('v_mfma_'):
            match=re.match(r'v_mfma_\S+ v\[(\d+):(\d+)\]',op)
            first,last=map(int,match.groups())
            require(first>=128 and last<=255,'Raw MFMA outside two banks')
            for register in range(first,last+1):
                owner,contributions=raw[register]
                if op.endswith(', 0'): owner,contributions=scale,0
                require(owner==scale,'Raw bank not reset at K128 boundary')
                raw[register]=(owner,contributions+1)
                last_write[register]=mfmas
            mfmas+=1
        elif opcode=='v_fma_f32':
            match=re.fullmatch(r'v_fma_f32 v(\d+), v(\d+), s(\d+), v(\d+)',op)
            destination,source,scalar,accumulator=map(int,match.groups())
            require(destination==accumulator and 64<=destination<128,'Wrong scaled bank')
            owner,contributions=raw[source]
            tag=previous if scalar==15 else scale
            require(scalar in (14,15) and owner==tag,'Scale/raw bank mismatch')
            if owner is not None:
                require(contributions==min(4,stages-owner*4)*4,'Partial group contribution count wrong')
                require(output[destination]==list(range(owner)),'Dropped/reordered/duplicate scale group')
                output[destination].append(owner)
                if scalar==15:
                    distance=mfmas-last_write[source]
                    minimum_distance=distance if minimum_distance is None else min(minimum_distance,distance)
        counts[opcode]+=1
    groups=(stages+3)//4
    require(all(value==list(range(groups)) for value in output.values()),'Final grouped sums incomplete')
    require(scale_loads==groups,'Wrong scale metadata load count')
    require(mfmas==16*stages,'Wrong raw MFMA count')
    require(counts['global_load_dwordx4']==3*stages,'Weight over/under-fetch')
    require(counts['ds_write_b128']==4*stages,'Wrong LDS writes')
    require(minimum_distance is None or minimum_distance>=8,'Too few MFMA issue intervals at previous-bank drain')
    return dict(stages=stages,groups=groups,mfmas=mfmas,minimum_prior_bank_issue_distance=minimum_distance,
                scalar_loads=scale_loads,scale_fmas=counts['v_fma_f32'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/scaled-audit')
    args=parser.parse_args()
    out=args.output.resolve()
    require(not out.exists() or not any(out.iterdir()),'Choose a fresh output')
    out.mkdir(parents=True,exist_ok=True)
    build=ROOT/'temp/native-gemm/scaled/scaled'
    code=asm_instructions((build/'compute.hpp').read_text())
    checks=[execute(code,count) for count in STAGE_COUNTS]
    a=elf_text(ROOT/'temp/native-gemm/scaled/compact/probe.hsaco')
    b=elf_text(ROOT/'temp/native-gemm/scaled/heavy/probe.hsaco')
    require(a==b,'Reserved-register control changed instruction bytes')
    proof=dict(checks=checks,compact_heavy_text_identical=True,text_bytes=len(a),
               text_sha256=hashlib.sha256(a).hexdigest())
    (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('PASS',len(checks),'stage counts: exact ordered K128 groups/banks/scales/no overfetch; compact/heavy whole.text identical')


if __name__=='__main__': main()
