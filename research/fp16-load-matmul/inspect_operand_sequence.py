#!/usr/bin/env python3
"""Symbolically execute generated asm to check record/word progression.
Loads are assumed instantly complete: this rules out an indexing error, NOT
hardware timing/readiness hazards. The real GPU oracle tests remain separate.
"""
import argparse,json,re
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
s=(a.folder/'pipeline.hpp').read_text()
d,u=map(int,re.search(r'kDepth=(\d+), kUnroll=(\d+)',s).groups())
stride=int(re.search(r'kLdsStride=(\d+)',s)[1])
results=[]
for mode in (1,2,4):
    for loads in (1,2,4):
        key=f'if constexpr(Mode=={mode} && Loads=={loads})'
        block=s.split(key,1)[1].split('      : [c]',1)[0]
        code=[json.loads(l.strip()).strip() for l in block.splitlines() if l.strip().startswith('"')]
        labels={l[:-1]:i for i,l in enumerate(code) if l.endswith(':')}
        steps=40 if u==8 else 4096
        records=steps+d+2 if mode==2 else (7 if u==8 else 256 if mode==1 else 16//stride)
        unit=1024*(stride if mode==4 else 1)
        scalar={'remaining':steps,'offset':0,'preleft':records,'window':records*unit,'blo':0,'bhi':0,'plo':0,'phi':0}
        vgpr={};pc=0;n=0;scc=False;executed=0
        def atom(x):
            x=x.strip()
            if x.startswith('%['):return scalar[x[2:-1]]
            if re.fullmatch(r'v\d+',x):return vgpr[int(x[1:])]
            return int(x)
        def vrange(x):
            x=x.strip().split()[0]
            if ':' in x:
                lo,hi=map(int,re.search(r'\[(\d+):(\d+)\]',x).groups());return list(range(lo,hi+1))
            return [int(x.strip()[1:])]
        while pc<len(code):
            line=code[pc];pc+=1;executed+=1
            if executed>1000000:raise RuntimeError('symbolic execution did not terminate')
            if line.endswith(':'):continue
            op,_,args=line.partition(' ')
            argv=[x.strip() for x in args.split(',')]
            if op in ('s_memtime','s_memrealtime','s_waitcnt','s_nop'):continue
            if op=='s_mov_b32':scalar[argv[0][2:-1]]=atom(argv[1]);continue
            if op in ('s_add_u32','s_sub_u32'):
                scalar[argv[0][2:-1]]=atom(argv[1])+(atom(argv[2]) if op=='s_add_u32' else -atom(argv[2]));continue
            if op.startswith('s_cmp_'):
                x,y=map(atom,argv);scc=x==y if op=='s_cmp_eq_u32' else x>=y if op=='s_cmp_ge_u32' else x!=y;continue
            if op=='s_cselect_b32':scalar[argv[0][2:-1]]=atom(argv[1] if scc else argv[2]);continue
            if op in ('s_cbranch_scc0','s_cbranch_scc1','s_branch'):
                if op=='s_branch' or scc==(op=='s_cbranch_scc1'):pc=labels[args]
                continue
            if op in ('v_add_u32','v_add_co_u32','v_addc_co_u32'):
                src=argv[2:4] if op!='v_add_u32' else argv[1:3]
                vgpr[int(argv[0][1:])]=sum(map(atom,src));continue
            if op=='v_mov_b32':vgpr[int(argv[0][1:])]=atom(argv[1]);continue
            if op.startswith(('global_load_','ds_read_')):
                dest=vrange(argv[0]);addr=atom('v'+str(vrange(argv[1])[0]))
                if (m:=re.search(r'offset:(\d+)',line)):addr+=int(m[1])
                rec,offset=divmod(addr,unit)
                # Lane zero, so all intra-record offsets select A/B word planes.
                word=0 if loads==1 else (2 if offset else 0) if loads==2 else offset//(256*(stride if mode==4 else 1))
                for i,v in enumerate(dest):vgpr[v]=(rec,word+i)
                continue
            if op=='v_mfma_f32_32x32x8f16':
                values=[vgpr[v] for x in argv[1:3] for v in vrange(x)]
                record=n if mode==2 else n%records
                assert values==[(record,i) for i in range(4)],(mode,loads,n,values,record)
                n+=1;continue
            raise RuntimeError('unhandled instruction '+line)
        assert n==steps
        results.append(dict(mode=mode,loads=loads,steps=steps,records=records,mfmas=n))
(a.folder/'operand-sequence.json').write_text(json.dumps(dict(depth=d,unroll=u,operand_storage='raw_ring',cases=results),indent=2)+'\n')
print(f'D{d}/U{u}/direct raw: 9 generated instruction streams supply exactly records 0..steps-1, words A0,A1,B0,B1 under ideal completion. No indexing error detected; no hardware hazard conclusion.')
