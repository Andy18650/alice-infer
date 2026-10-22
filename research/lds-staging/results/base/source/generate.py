#!/usr/bin/env python3
"""One64-load loop: direct-to-LDS DWORD, no wait/barrier/consumer inside."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ops=['s_mov_b32 %[saved], m0',
     's_memtime %[begin]','s_memrealtime %[wall_begin]','s_waitcnt lgkmcnt(0)',
     '.Lstaging_loop_%=:']
for group in range(4):
    ops.append(f's_add_u32 m0, %[dst], {group*4096}')
    if group==0:
        ops+=['v_add_co_u32 v8, vcc, %[offset], %[lo]',
              'v_addc_co_u32 v9, vcc, 0, %[hi], vcc']
    else:
        ops+=['v_add_co_u32 v8, vcc, 4096, v8','v_addc_co_u32 v9, vcc, 0, v9, vcc']
    # Immediate displacement changes global source AND LDS destination.
    for i in range(16):
        off=f' offset:{256*i}' if i else ''
        ops.append('global_load_dword v[8:9], off'+off+' lds')
ops+=['s_add_u32 %[offset], %[offset], 16384',
      's_and_b32 %[offset], %[offset], 262143', #256KiB/wave,1MiB aggregate
      's_sub_u32 %[remaining], %[remaining], 1','s_cmp_lg_u32 %[remaining], 0',
      's_cbranch_scc1 .Lstaging_loop_%=',
      's_memtime %[issued]','s_memrealtime %[wall_issued]','s_waitcnt lgkmcnt(0)',
      # Terminal completion only, never a periodic backpressure policy.
      's_waitcnt vmcnt(0)',
      's_memtime %[completed]','s_memrealtime %[wall_completed]','s_waitcnt lgkmcnt(0)',
      's_mov_b32 m0, %[saved]']
text=['__device__ __forceinline__ void load_loop(uint32_t lo,uint32_t hi,uint32_t dst,uint32_t groups,',
      '    uint64_t& begin,uint64_t& issued,uint64_t& completed,',
      '    uint64_t& wall_begin,uint64_t& wall_issued,uint64_t& wall_completed) {',
      '  uint32_t remaining=groups,offset=0,saved;',
      '  asm volatile(']+['    "'+op+'\\n\\t"' for op in ops]+[
      '    : [begin] "=&s"(begin),[issued] "=&s"(issued),[completed] "=&s"(completed),',
      '      [wall_begin] "=&s"(wall_begin),[wall_issued] "=&s"(wall_issued),[wall_completed] "=&s"(wall_completed),',
      '      [remaining] "+&s"(remaining),[offset] "+&s"(offset),[saved] "=&s"(saved)',
      '    : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst)',
      '    : "v8","v9","vcc","scc","memory");','}']
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(text)+'\n')
