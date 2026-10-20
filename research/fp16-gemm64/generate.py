#!/usr/bin/env python3
"""Two independently addressed shared A/B LDS streams, direct into MFMA."""
import argparse
from math import gcd
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=2);p.add_argument('--unroll',type=int,default=4);p.add_argument('--stage-k',type=int,default=64);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();D,U,T=a.depth,a.unroll,a.stage_k
if not 1<=D<=8 or T not in (32,64,128) or U<1 or (T//8)%U:p.error('depth1..8; stage-k32/64/128; unroll must divide stage-k/8')
P=D//gcd(D,U);va=4*D;vb=va+1
out=[f'constexpr int kDepth={D}, kUnroll={U}, kStageK={T}, kPhases={P};',
     'template<bool Hot> __device__ __forceinline__ void compute(uint32_t alo,uint32_t blo,int steps,int records,F16& c) {',
     '  uint32_t remaining=steps,offset=0; const uint32_t window=records*2048u;']
for hot in (True,False):
    lines=[];tag='hot' if hot else 'sweep'
    def emit(x):lines.append(x)
    def label(x):return f'.Lgemm64_{tag}_{x}_%='
    def addresses():
        emit(f'v_add_u32 v{va}, %[alo], %[offset]');emit(f'v_add_u32 v{vb}, %[blo], %[offset]')
    def issue(slot):
        if not hot:addresses()
        v=slot*4
        emit(f'ds_read_b64 v[{v}:{v+1}], v{va}')
        emit(f'ds_read_b64 v[{v+2}:{v+3}], v{vb}')
        if not hot:
            emit('s_add_u32 %[offset], %[offset], 2048');emit('s_cmp_ge_u32 %[offset], %[window]');emit('s_cselect_b32 %[offset], 0, %[offset]')
    if hot:addresses()
    for n in range(D-1):issue(n)
    for phase in range(P):
        emit(label(str(phase))+':')
        for j in range(U):
            n=phase*U+j;issue((n+D-1)%D)
            emit(f's_waitcnt lgkmcnt({2*(D-1)})')
            v=4*(n%D);emit(f'v_mfma_f32_32x32x8f16 %[c], v[{v}:{v+1}], v[{v+2}:{v+3}], %[c]')
        emit(f's_sub_u32 %[remaining], %[remaining], {U}');emit('s_cmp_eq_u32 %[remaining], 0')
        emit('s_cbranch_scc0 '+label(str((phase+1)%P)))
        if P>1:emit('s_branch '+label('done'))
    emit(label('done')+':');emit('s_waitcnt lgkmcnt(0)')
    out+=['  if constexpr(Hot=='+str(hot).lower()+') {','    asm volatile(']
    out+=['      "'+line+'\\n\\t"' for line in lines]
    out+=['      : [c] "+a"(c),[remaining] "+&s"(remaining),[offset] "+&s"(offset)',
          '      : [alo] "v"(alo),[blo] "v"(blo),[window] "s"(window)',
          '      : "memory","scc",'+','.join(f'"v{i}"' for i in range(vb+1))+');','  }']
out+=['}']
# Batched cooperative producers. Every pending VGPR in the register path
# stays in one asm through its vmcnt wait and first DS-write consumption.
G=T//16
for direct in (False,True):
    lines=[]
    if direct:
        lines=['s_mov_b32 %[saved], m0','s_mov_b32 m0, %[dst]','s_nop 0',
               'v_mov_b32 v0, %[lo]','v_mov_b32 v1, %[hi]']
        for g in range(G):
            for off in (0,256,2048,2304):
                lines.append('global_load_dword v[0:1], off'+(f' offset:{off}' if off else '')+' lds')
            if g+1<G:
                lines+=['v_add_co_u32 v0, vcc, 4096, v0','v_addc_co_u32 v1, vcc, 0, v1, vcc',
                        's_add_u32 m0, m0, 4096','s_nop 0']
        lines+=['s_mov_b32 m0, %[saved]']
        clobbers=['v0','v1','vcc','scc','memory']
        sig='void copy_direct(uint32_t lo,uint32_t hi,uint32_t dst) {'
        output='[saved] "=&s"(saved)';inputs='[lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst)'
    else:
        p=4*G;ld=p+2
        lines=[f'v_mov_b32 v{p}, %[lo]',f'v_mov_b32 v{p+1}, %[hi]',f'v_mov_b32 v{ld}, %[dst]']
        for g in range(G):
            lines.append(f'global_load_dwordx4 v[{4*g}:{4*g+3}], v[{p}:{p+1}], off')
            if g+1<G:lines +=[f'v_add_co_u32 v{p}, vcc, 4096, v{p}',f'v_addc_co_u32 v{p+1}, vcc, 0, v{p+1}, vcc']
        lines+=['s_waitcnt vmcnt(0)']
        for g in range(G):lines.append(f'ds_write_b128 v{ld}, v[{4*g}:{4*g+3}]'+(f' offset:{g*4096}' if g else ''))
        clobbers=[*(f'v{i}' for i in range(ld+1)),'vcc','memory']
        sig='void copy_reg(uint32_t lo,uint32_t hi,uint32_t dst) {'
        output='';inputs='[lo] "v"(lo),[hi] "v"(hi),[dst] "v"(dst)'
    out+=['__device__ __forceinline__ '+sig]
    if direct:out+=['  uint32_t saved;']
    out+=['  asm volatile(']+['    "'+line+'\\n\\t"' for line in lines]
    out+=['    : '+output+' : '+inputs+' : '+','.join('"'+c+'"' for c in clobbers)+');','}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(out)+'\n')
print(f'Generated two-stream LDS compute D={D} U={U} stageK={T}, phases={P}, no copies/ready banks.')
