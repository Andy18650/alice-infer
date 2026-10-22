#!/usr/bin/env python3
"""64 register-resident per-wave timing samples, no result stores inside."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ops=['s_mov_b32 %[saved], m0']
for group in range(4):
    ops.append(f's_add_u32 m0, %[dst], {group*4096}')
    if group==0:
        ops+=['v_mov_b32 v8, %[lo]','v_mov_b32 v9, %[hi]']
    else:
        ops+=['v_add_co_u32 v8, vcc, 4096, v8','v_addc_co_u32 v9, vcc, 0, v9, vcc']
    for i in range(16):
        n=group*16+i;off=f' offset:{256*i}' if i else ''
        ops+=['s_memtime s[16:17]','s_waitcnt lgkmcnt(0)',
              'global_load_dword v[8:9], off'+off+' lds','s_waitcnt vmcnt(0)',
              's_memtime s[18:19]','s_waitcnt lgkmcnt(0)',
              's_sub_u32 %[delta], s18, s16',f'v_mov_b32 %[r{n}], %[delta]']
ops+=['s_mov_b32 m0, %[saved]']
text=['constexpr int kSamples=64;',
      '__device__ __forceinline__ void measure(uint32_t lo,uint32_t hi,uint32_t dst,uint32_t (&samples)[kSamples]) {',
      '  uint32_t delta,saved;',
      '  asm volatile(']+['    "'+op+'\\n\\t"' for op in ops]
text+=['    : '+',\n      '.join(f'[r{i}] "=&v"(samples[{i}])' for i in range(64))+',',
       '      [delta] "=&s"(delta),[saved] "=&s"(saved)',
       '    : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst)',
       '    : "v8","v9","s16","s17","s18","s19","vcc","scc","memory");','}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(text)+'\n')
