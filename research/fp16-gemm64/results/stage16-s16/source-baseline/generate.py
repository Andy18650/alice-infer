#!/usr/bin/env python3
"""Fixed winning K8/16-stage/raw2 GEMM plus optional LDS diagnostics."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
D,U,T,N=2,1,8,16
P,va,vb=2,8,9
out=['constexpr int kDepth=2, kUnroll=1, kStageK=8, kPhases=2, kStages=16;',
     'template<bool Hot> __device__ __forceinline__ void compute(uint32_t alo,uint32_t blo,int steps,int records,F16& c) {',
     '  uint32_t remaining=steps,offset=0; const uint32_t window=records*2048u;']
# Keep the independent math-only diagnostics unchanged.
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
        emit('s_branch '+label('done'))
    emit(label('done')+':');emit('s_waitcnt lgkmcnt(0)')
    out+=['  if constexpr(Hot=='+str(hot).lower()+') {','    asm volatile(']
    out+=['      "'+line+'\\n\\t"' for line in lines]
    out+=['      : [c] "+a"(c),[remaining] "+&s"(remaining),[offset] "+&s"(offset)',
          '      : [alo] "v"(alo),[blo] "v"(blo),[window] "s"(window)',
          '      : "memory","scc",'+','.join(f'"v{i}"' for i in range(vb+1))+');','  }']
out+=['}']
# Test-only one-record transfer, independent of the GEMM's paired addressing.
out+=['__device__ __forceinline__ void copy_direct(uint32_t lo,uint32_t hi,uint32_t dst) {',
      '  uint32_t saved;', '  asm volatile(',
      '    "s_mov_b32 %[saved], m0\\n\\t"',
      '    "s_mov_b32 m0, %[dst]\\n\\t"',
      '    "s_nop 0\\n\\t"',
      '    "v_mov_b32 v10, %[lo]\\n\\t"',
      '    "v_mov_b32 v11, %[hi]\\n\\t"',
      '    "global_load_dword v[10:11], off lds\\n\\t"',
      '    "global_load_dword v[10:11], off offset:256 lds\\n\\t"',
      '    "s_mov_b32 m0, %[saved]\\n\\t"',
      '    : [saved] "=&s"(saved) : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst)',
      '    : "v10","v11","vcc","scc","memory");','}']
out+=['__device__ __forceinline__ void pipeline(uint32_t lo,uint32_t hi,uint32_t dst,uint32_t alo,uint32_t blo,int tiles,F16& c) {',
      '  uint32_t remaining=tiles,saved;']
lines=[]
def emit(x):lines.append(x)
def lab(x):return f'.Lpipe8_16_{x}_%='
def branch(x):emit('s_branch '+lab(x))
def produce(slot,prepared=False):
    displacement=(slot%2)*2048
    if not prepared:
        emit(f's_add_u32 m0, %[dst], {(slot//2)*4096}');emit('s_nop 0')
    for off in (displacement,displacement+256):
        emit('global_load_dword v[8:9], off'+(f' offset:{off}' if off else '')+' lds')
    emit('s_mov_b32 m0, %[saved]')
    if slot%2:
        emit('v_add_co_u32 v8, vcc, 4096, v8');emit('v_addc_co_u32 v9, vcc, 0, v9, vcc')
def read(slot,raw):
    v=4*raw;off=f' offset:{slot*2048}' if slot else ''
    emit(f'ds_read_b64 v[{v}:{v+1}], %[alo]'+off)
    emit(f'ds_read_b64 v[{v+2}:{v+3}], %[blo]'+off)
def mfma(raw,tag=None):
    if tag:
        emit('s_cmp_eq_u32 %[remaining], 1');emit('s_cbranch_scc0 '+lab(tag+'_wait2'))
        emit('s_waitcnt lgkmcnt(0)');branch(tag+'_ready');emit(lab(tag+'_wait2')+':')
    emit('s_waitcnt lgkmcnt(2)')
    if tag:emit(lab(tag+'_ready')+':')
    v=4*raw;emit(f'v_mfma_f32_32x32x8f16 %[c], v[{v}:{v+1}], v[{v+2}:{v+3}], %[c]')
def wait_tail(maxlater,shift,tag,drain_short=False):
    for later in range(maxlater,-1,-1):
        emit(f's_cmp_ge_u32 %[remaining], {later+shift}');emit('s_cbranch_scc1 '+lab(f'{tag}_{later}'))
    if drain_short:emit('s_waitcnt vmcnt(0)')
    branch(tag+'_done')
    for later in range(maxlater,-1,-1):
        emit(lab(f'{tag}_{later}')+':');emit(f's_waitcnt vmcnt({2*later})');branch(tag+'_done')
    emit(lab(tag+'_done')+':')
emit('s_mov_b32 %[saved], m0');emit('s_cmp_eq_u32 %[remaining], 0');emit('s_cbranch_scc1 '+lab('done'))
emit('v_mov_b32 v8, %[lo]');emit('v_mov_b32 v9, %[hi]')
for slot in range(N):
    emit(f's_cmp_le_u32 %[remaining], {slot}');emit('s_cbranch_scc1 '+lab('primed'));produce(slot)
emit(lab('primed')+':');wait_tail(N-1,1,'initial0');emit('s_barrier');read(0,0)
wait_tail(N-3,3,'initial2',drain_short=True);emit('s_barrier')
emit('s_cmp_le_u32 %[remaining], 1');emit('s_cbranch_scc1 '+lab('second_prime_done'));read(1,1)
emit(lab('second_prime_done')+':');emit(f's_cmp_le_u32 %[remaining], {2*N}');emit('s_cbranch_scc1 '+lab('tail0'))
emit(lab('bulk')+':')
for phase in range(N):
    raw=phase%2;mfma(raw);read((phase+2)%N,raw);emit('s_waitcnt vmcnt(24)')
    emit(f's_add_u32 m0, %[dst], {(phase//2)*4096}');emit('s_barrier');produce(phase,prepared=True)
emit('s_sub_u32 %[remaining], %[remaining], 16');emit('s_cmp_gt_u32 %[remaining], 32');emit('s_cbranch_scc1 '+lab('bulk'))
for phase in range(N):
    raw=phase%2;tag=f'tail{phase}';emit(lab(tag)+':');mfma(raw,tag)
    emit('s_cmp_le_u32 %[remaining], 2');emit('s_cbranch_scc1 '+lab(tag+'_read_done'))
    read((phase+2)%N,raw);emit(lab(tag+'_read_done')+':');wait_tail(N-4,4,tag+'_publish');emit('s_barrier')
    emit(f's_cmp_le_u32 %[remaining], {N}');emit('s_cbranch_scc1 '+lab(tag+'_refill_done'));produce(phase)
    emit(lab(tag+'_refill_done')+':');emit('s_sub_u32 %[remaining], %[remaining], 1')
    emit('s_cmp_eq_u32 %[remaining], 0');emit('s_cbranch_scc1 '+lab('done'))
branch('tail0');emit(lab('done')+':');emit('s_mov_b32 m0, %[saved]')
out+=['  asm volatile(']+['    "'+line+'\\n\\t"' for line in lines]
out+=['    : [c] "+a"(c),[remaining] "+&s"(remaining),[saved] "=&s"(saved)',
      '    : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst),[alo] "v"(alo),[blo] "v"(blo)',
      '    : "memory","scc","vcc",'+','.join(f'"v{i}"' for i in range(10))+');','}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(out)+'\n')
print('Generated fixed K8 /16 LDS slots /raw2 /outer-unroll16; two LDS-only diagnostics.')
