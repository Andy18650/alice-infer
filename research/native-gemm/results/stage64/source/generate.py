#!/usr/bin/env python3
"""Native MFMA square CTA128/256,K32/64,single LDS,Q+1 raw slots."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tile',type=int,choices=(128,256),default=128);p.add_argument('--output',type=Path,required=True);p.add_argument('--resident',action='store_true');p.add_argument('--stage-k',type=int,choices=(32,64),default=64);a=p.parse_args()
T=a.tile;S=a.stage_k;R=T//64;P=R//2;Q=S//8;raw_width=4*R
ptr=(Q+1)*raw_width;gbase=(ptr+2+3)//4*4;packets=Q*P;last=gbase+packets*4
ops=[]
def emit(s):ops.append(s)
def label(s):return f'.Lnative_{T}_{S}_{s}_%='
def branch(s):emit('s_branch '+label(s))
def gl():
    for i in range(packets):
        if i and i%4==0:
            emit(f'v_add_co_u32 v{ptr}, vcc, 4096, v{ptr}');emit(f'v_addc_co_u32 v{ptr+1}, vcc, 0, v{ptr+1}, vcc')
        off=(i%4)*1024;v=gbase+4*i
        emit(f'global_load_dwordx4 v[{v}:{v+3}], v[{ptr}:{ptr+1}], off'+(f' offset:{off}' if off else ''))
    emit(f'v_add_co_u32 v{ptr}, vcc, %[step], v{ptr}');emit(f'v_addc_co_u32 v{ptr+1}, vcc, 0, v{ptr+1}, vcc')
def write():
    for q in range(Q):
        for pair in range(P):
            off=q*T*16+pair*1024;v=gbase+4*(q*P+pair)
            emit(f'ds_write_b128 %[dst], v[{v}:{v+3}]'+(f' offset:{off}' if off else ''))
def read(q,slot):
    for kind,addr in ((0,'alo'),(1,'blo')):
        for pair in range(P):
            off=q*T*16+pair*1024;v=slot*raw_width+kind*2*R+pair*4
            emit(f'ds_read_b128 v[{v}:{v+3}], %[{addr}]'+(f' offset:{off}' if off else ''))
def math(slot):
    b=slot*raw_width
    for i in range(R):
        for j in range(R):
            x=b+2*i;y=b+2*R+2*j;c=i*R+j
            emit(f'v_mfma_f32_32x32x8f16 %[c{c}], v[{x}:{x+1}], v[{y}:{y+1}], %[c{c}]')
emit('s_memtime %[begin]');emit('s_memrealtime %[wall_begin]');emit('s_waitcnt lgkmcnt(0)')
emit('s_cmp_eq_u32 %[remaining], 0');emit('s_cbranch_scc1 '+label('done'))
emit(f'v_mov_b32 v{ptr}, %[lo]');emit(f'v_mov_b32 v{ptr+1}, %[hi]')
gl();emit('s_waitcnt vmcnt(0)');write();emit('s_waitcnt lgkmcnt(0)');emit('s_barrier')
emit('s_cmp_le_u32 %[remaining], 1');emit('s_cbranch_scc1 '+label('seed'));gl()
emit(label('seed')+':')
for q in range(Q):read(q,q)
if a.resident:
    emit('s_waitcnt lgkmcnt(0)');emit(label('resident')+':')
    for q in range(Q):math(q)
    emit('s_sub_u32 %[remaining], %[remaining], 1');emit('s_cmp_lg_u32 %[remaining], 0');emit('s_cbranch_scc1 '+label('resident'))
else:
    for phase in range(2):
        current=Q-1+phase;spare=Q-phase
        emit(label(f'phase{phase}')+':');emit('s_cmp_eq_u32 %[remaining], 1');emit('s_cbranch_scc1 '+label(f'tail{phase}'))
        # Retire all current-buffer readers BEFORE overwriting; then stage next.
        emit('s_waitcnt lgkmcnt(0)');emit('s_barrier');emit('s_waitcnt vmcnt(0)');write()
        emit('s_cmp_le_u32 %[remaining], 2');emit('s_cbranch_scc1 '+label(f'no_future{phase}'));gl()
        emit(label(f'no_future{phase}')+':')
        publish=Q//4
        for q in range(publish):math(q)
        # Publish after first quarter; roll reads behind consumed fragments.
        emit('s_waitcnt lgkmcnt(0)');emit('s_barrier')
        for q in range(publish):read(q,q)
        read(Q-1,spare)
        for q in range(publish,Q-1):math(q);read(q,q)
        math(current)
        emit('s_sub_u32 %[remaining], %[remaining], 1');branch(f'phase{1-phase}')
    for phase in range(2):
        emit(label(f'tail{phase}')+':');emit('s_waitcnt lgkmcnt(0)')
        for q in range(Q-1):math(q)
        math(Q-1+phase);branch('done')
emit(label('done')+':')
# Constant conservative matrix-tail completion gap, negligible for long K.
for _ in range(4):emit('s_nop 15')
emit('s_memtime %[end]');emit('s_memrealtime %[wall_end]');emit('s_waitcnt lgkmcnt(0)')
text=[f'constexpr int kTile={T},kR={R},kStageK={S},kQ={Q},kLdsBytes={T*S*4};',f'constexpr bool kResident={str(a.resident).lower()};',
      '__device__ __forceinline__ void pipeline(uint32_t lo,uint32_t hi,uint32_t step,uint32_t dst,uint32_t alo,uint32_t blo,int tiles,F16 (&c)[kR*kR],uint64_t& begin,uint64_t& end,uint64_t& wall_begin,uint64_t& wall_end) {',
      '  uint32_t remaining=tiles;', '  asm volatile(']+['    "'+s+'\\n\\t"' for s in ops]
text+=['    : '+',\n      '.join(f'[c{i}] "+a"(c[{i}])' for i in range(R*R))+',',
       '      [remaining] "+&s"(remaining),[begin] "=&s"(begin),[end] "=&s"(end),[wall_begin] "=&s"(wall_begin),[wall_end] "=&s"(wall_end)',
       '    : [lo] "v"(lo),[hi] "v"(hi),[step] "s"(step),[dst] "v"(dst),[alo] "v"(alo),[blo] "v"(blo)',
       '    : "memory","scc","vcc",'+','.join(f'"v{i}"' for i in range(last))+');','}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(text)+'\n')
