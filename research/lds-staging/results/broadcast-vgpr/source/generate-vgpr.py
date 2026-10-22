#!/usr/bin/env python3
"""32 x4-VGPR slots, one opaque continuously overwritten load ring."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ops=['s_memtime %[begin]','s_memrealtime %[wall_begin]','s_waitcnt lgkmcnt(0)',
     '.Lvgpr_ring_%=:']
for group in range(8):
    if group==0:ops+=['v_add_co_u32 v8, vcc, %[offset], %[lo]','v_addc_co_u32 v9, vcc, 0, %[hi], vcc']
    else:ops+=['v_add_co_u32 v8, vcc, 4096, v8','v_addc_co_u32 v9, vcc, 0, v9, vcc']
    for j in range(4):
        off=f' offset:{j*1024}' if j else ''
        ops.append(f'global_load_dwordx4 %[r{group*4+j}], v[8:9], off'+off)
ops+=['s_add_u32 %[offset], %[offset], 32768','s_and_b32 %[offset], %[offset], %[mask]',
      's_sub_u32 %[remaining], %[remaining], 1','s_cmp_lg_u32 %[remaining], 0','s_cbranch_scc1 .Lvgpr_ring_%=',
      's_memtime %[issued]','s_memrealtime %[wall_issued]','s_waitcnt lgkmcnt(0)',
      's_waitcnt vmcnt(0)',
      's_memtime %[completed]','s_memrealtime %[wall_completed]','s_waitcnt lgkmcnt(0)']
text=['constexpr int kRing=32;',
      '__device__ __forceinline__ void load_vgpr(uint32_t lo,uint32_t hi,uint32_t mask,uint32_t groups,U4 (&ring)[kRing],',
      '    uint64_t& begin,uint64_t& issued,uint64_t& completed,',
      '    uint64_t& wall_begin,uint64_t& wall_issued,uint64_t& wall_completed) {',
      '  uint32_t remaining=groups,offset=0;',
      '  asm volatile(']+['    "'+op+'\\n\\t"' for op in ops]
text+=['    : '+',\n      '.join(f'[r{i}] "=&v"(ring[{i}])' for i in range(32))+',',
       '      [begin] "=&s"(begin),[issued] "=&s"(issued),[completed] "=&s"(completed),',
       '      [wall_begin] "=&s"(wall_begin),[wall_issued] "=&s"(wall_issued),[wall_completed] "=&s"(wall_completed),',
       '      [remaining] "+&s"(remaining),[offset] "+&s"(offset)',
       '    : [lo] "v"(lo),[hi] "v"(hi),[mask] "s"(mask)',
       '    : "v8","v9","vcc","scc","memory");','}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(text)+'\n')
