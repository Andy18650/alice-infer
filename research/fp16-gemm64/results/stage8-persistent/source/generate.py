#!/usr/bin/env python3
"""Two independently addressed shared A/B LDS streams, direct into MFMA."""
import argparse
from math import gcd
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=2);p.add_argument('--unroll',type=int,default=4);p.add_argument('--stage-k',type=int,default=64);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();D,U,T=a.depth,a.unroll,a.stage_k
if not 1<=D<=8 or T not in (8,32,64,128) or U<1 or (T//8)%U:p.error('depth1..8; stage-k8/32/64/128; unroll must divide stage-k/8')
if T==8 and D!=2:p.error('stage-k8 persistent pipeline uses exactly two raw register buffers')
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
# Direct-to-LDS only. m0 is saved/restored within one producer asm block.
G=T//16
lines=['s_mov_b32 %[saved], m0','s_mov_b32 m0, %[dst]','s_nop 0',
       'v_mov_b32 v0, %[lo]','v_mov_b32 v1, %[hi]']
for g in range(G):
    for off in (0,256,2048,2304):
        lines.append('global_load_dword v[0:1], off'+(f' offset:{off}' if off else '')+' lds')
    if g+1<G:
        lines+=['v_add_co_u32 v0, vcc, 4096, v0','v_addc_co_u32 v1, vcc, 0, v1, vcc',
                's_add_u32 m0, m0, 4096','s_nop 0']
if T==8:
    # One K8 record; keep producer addresses outside every MFMA raw slot.
    ptr=4*D+2
    lines=['s_mov_b32 %[saved], m0','s_mov_b32 m0, %[dst]','s_nop 0',
           f'v_mov_b32 v{ptr}, %[lo]',f'v_mov_b32 v{ptr+1}, %[hi]',
           f'global_load_dword v[{ptr}:{ptr+1}], off lds',
           f'global_load_dword v[{ptr}:{ptr+1}], off offset:256 lds']
else:ptr=0
lines+=['s_mov_b32 m0, %[saved]']
out+=['__device__ __forceinline__ void copy_direct(uint32_t lo,uint32_t hi,uint32_t dst) {',
      '  uint32_t saved;','  asm volatile(']
out+=['    "'+line+'\\n\\t"' for line in lines]
out+=['    : [saved] "=&s"(saved) : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst)',
      f'    : "v{ptr}","v{ptr+1}","vcc","scc","memory");','}']
if T!=8:
    out+=['template<int Stages> __device__ __forceinline__ void pipeline(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int,F16&);']
if T==8:
    out+=['template<int Stages> __device__ __forceinline__ void pipeline(uint32_t lo,uint32_t hi,uint32_t dst,uint32_t alo,uint32_t blo,int tiles,F16& c) {',
          '  uint32_t remaining=tiles,saved;']
    for N in (2,4,8):
        lines=[]
        def emit(x):lines.append(x)
        def lab(x):return f'.Lpipe8_{N}_{x}_%='
        def branch(x):emit('s_branch '+lab(x))
        def produce(slot):
            emit(f's_add_u32 m0, %[dst], {slot*2048}')
            emit('s_nop 0')
            emit('global_load_dword v[8:9], off lds')
            emit('global_load_dword v[8:9], off offset:256 lds')
            emit('s_mov_b32 m0, %[saved]')
            emit('v_add_co_u32 v8, vcc, 2048, v8')
            emit('v_addc_co_u32 v9, vcc, 0, v9, vcc')
        def read(slot,raw):
            v=4*raw;off=f' offset:{slot*2048}' if slot else ''
            emit(f'ds_read_b64 v[{v}:{v+1}], %[alo]'+off)
            emit(f'ds_read_b64 v[{v+2}:{v+3}], %[blo]'+off)
        def mfma(raw):
            v=4*raw
            emit(f'v_mfma_f32_32x32x8f16 %[c], v[{v}:{v+1}], v[{v+2}:{v+3}], %[c]')
            # Prefetch overlaps the current MFMA; complete it before retiring
            # its LDS slot, so its register values survive an early refill.
            emit('s_waitcnt lgkmcnt(0)')
        def wait_tail(maxlater,shift,tag):
            for later in range(maxlater,-1,-1):
                emit(f's_cmp_ge_u32 %[remaining], {later+shift}')
                emit('s_cbranch_scc1 '+lab(f'{tag}_{later}'))
            branch(tag+'_done')
            for later in range(maxlater,-1,-1):
                emit(lab(f'{tag}_{later}')+':')
                emit(f's_waitcnt vmcnt({2*later})');branch(tag+'_done')
            emit(lab(tag+'_done')+':')
        emit('s_mov_b32 %[saved], m0')
        emit('s_cmp_eq_u32 %[remaining], 0');emit('s_cbranch_scc1 '+lab('done'))
        emit('v_mov_b32 v8, %[lo]');emit('v_mov_b32 v9, %[hi]')
        for slot in range(N):
            emit(f's_cmp_le_u32 %[remaining], {slot}')
            emit('s_cbranch_scc1 '+lab('primed'));produce(slot)
        emit(lab('primed')+':')
        wait_tail(N-1,1,'initial0');emit('s_barrier')
        read(0,0);emit('s_waitcnt lgkmcnt(0)')
        wait_tail(N-2,2,'initial1');emit('s_barrier')
        emit(f's_cmp_le_u32 %[remaining], {N}')
        emit('s_cbranch_scc1 '+lab('initial_refill_done'));produce(0)
        emit(lab('initial_refill_done')+':')
        emit(f's_cmp_le_u32 %[remaining], {2*N}')
        emit('s_cbranch_scc1 '+lab('tail0'))
        emit(lab('bulk')+':')
        for phase in range(N):
            slot=(phase+1)%N;raw=phase%2
            read(slot,1-raw);mfma(raw)
            emit(f's_waitcnt vmcnt({2*(N-2)})');emit('s_barrier');produce(slot)
        emit(f's_sub_u32 %[remaining], %[remaining], {N}')
        emit(f's_cmp_gt_u32 %[remaining], {2*N}')
        emit('s_cbranch_scc1 '+lab('bulk'))
        for phase in range(N):
            slot=(phase+1)%N;raw=phase%2;tag=f'tail{phase}'
            emit(lab(tag)+':')
            emit('s_cmp_le_u32 %[remaining], 1')
            emit('s_cbranch_scc1 '+lab(tag+'_read_done'))
            read(slot,1-raw);emit(lab(tag+'_read_done')+':');mfma(raw)
            wait_tail(N-2,3,tag+'_publish');emit('s_barrier')
            emit(f's_cmp_le_u32 %[remaining], {N+1}')
            emit('s_cbranch_scc1 '+lab(tag+'_refill_done'));produce(slot)
            emit(lab(tag+'_refill_done')+':')
            emit('s_sub_u32 %[remaining], %[remaining], 1')
            emit('s_cmp_eq_u32 %[remaining], 0');emit('s_cbranch_scc1 '+lab('done'))
        branch('tail0')
        emit(lab('done')+':');emit('s_mov_b32 m0, %[saved]')
        out+=['  if constexpr(Stages=='+str(N)+') {','    asm volatile(']
        out+=['      "'+line+'\\n\\t"' for line in lines]
        out+=['      : [c] "+a"(c),[remaining] "+&s"(remaining),[saved] "=&s"(saved)',
              '      : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst),[alo] "v"(alo),[blo] "v"(blo)',
              '      : "memory","scc","vcc",'+','.join(f'"v{i}"' for i in range(10))+');','  }']
    out+=['}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(out)+'\n')
print(f'Generated two-stream LDS compute D={D} U={U} stageK={T}, phases={P}, no copies/ready banks.')
