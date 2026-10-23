#!/usr/bin/env python3
"""128x128x32 W8A16: every wave produces one K8 slice of both operands.

A: two contiguous x4 loads/wave. B: one x4, expanded in place into eight
FP16 dwords. LDS layout, raw operands, MFMA order and peeled control flow
come from the measured native FP16 emitter. No wave-role branches.
"""

import argparse
from pathlib import Path

from generate import Configuration, PipelineEmitter


def conversion_ops(low, high, source):
    """Four operations/output pair; high first permits in-place expansion."""
    result = []
    for destination, selector in ((high, 'sel_hi'), (low, 'sel_lo')):
        result.extend([
            f'v_perm_b32 {destination}, 0, {source}, %[{selector}]',
            f'v_pk_ashrrev_i16 {destination}, 1, {destination} op_sel_hi:[0,1]',
            f'v_and_b32 {destination}, %[mask], {destination}',
            f'v_pk_mul_f16 {destination}, {destination}, %[scale]',
        ])
    return result


def block_conversion_ops():
    result = []
    for word in reversed(range(4)):
        result.extend(conversion_ops(
            f'v{52 + word * 2}', f'v{53 + word * 2}', f'v{52 + word}'))
    return result


CONSTANTS = ('[sel_lo] "s"(0x01040004u),[sel_hi] "s"(0x03040204u),'
             '[mask] "s"(0xbfffbfffu),[scale] "s"(0x5c005c00u)')


class Fp8Emitter(PipelineEmitter):
    def __init__(self):
        super().__init__(Configuration(128, 32))
        self.initialized_b_pointer = False

    def prefetch_global_stage(self):
        if not self.initialized_b_pointer:
            self.emit('v_mov_b32 v42, %[b_lo]')
            self.emit('v_mov_b32 v43, %[b_hi]')
            self.initialized_b_pointer = True
        for packet in range(2):
            register = 44 + packet * 4
            displacement = f' offset:{packet * 1024}' if packet else ''
            self.emit(f'global_load_dwordx4 v[{register}:{register + 3}], '
                      f'v[40:41], off{displacement}')
        self.emit('global_load_dwordx4 v[52:55], v[42:43], off')
        self.advance_source_pointer('%[step]')
        self.emit('v_add_co_u32 v42, vcc, %[b_step], v42')
        self.emit('v_addc_co_u32 v43, vcc, 0, v43, vcc')

    def write_prefetched_stage_to_lds(self):
        # A can start draining while the VALUs expand B. No producer register
        # is reloaded until all four LDS writes have issued.
        self.emit('ds_write_b128 %[dst], v[44:47]')
        self.emit('ds_write_b128 %[dst], v[48:51] offset:1024')
        # Backwards expansion protects unread packed source words. For word0,
        # high output must precede low output (the latter overwrites source0).
        for instruction in block_conversion_ops():
            self.emit(instruction)
        self.emit('ds_write_b128 %[b_dst], v[52:55]')
        self.emit('ds_write_b128 %[b_dst], v[56:59] offset:1024')

    def render(self):
        self.emit_startup()
        self.emit_gemm_loop_and_tails()
        self.emit_completion()
        lines = [
            'constexpr int kTile=128,kR=2,kStageK=32,kQ=4,kLdsBytes=16384;',
            'constexpr bool kResident=false,kFp8BStorage=true;',
            '__device__ __forceinline__ void pipeline(',
            '    uint32_t lo,uint32_t hi,uint32_t step,uint32_t dst,uint32_t alo,uint32_t blo,',
            '    uint32_t b_lo,uint32_t b_hi,uint32_t b_step,uint32_t b_dst,',
            '    int tiles,F16 (&c)[4],uint64_t& begin,uint64_t& end,',
            '    uint64_t& wall_begin,uint64_t& wall_end) {',
            '  uint32_t remaining=tiles;',
            '  asm volatile(',
        ]
        lines.extend('    "' + op + '\\n\\t"' for op in self.instructions)
        accumulators = ','.join(f'[c{i}] "+a"(c[{i}])' for i in range(4))
        clobbers = ','.join(f'"v{i}"' for i in range(60))
        lines.extend([
            '    : ' + accumulators + ',',
            '      [remaining] "+&s"(remaining),[begin] "=&s"(begin),[end] "=&s"(end),',
            '      [wall_begin] "=&s"(wall_begin),[wall_end] "=&s"(wall_end)',
            '    : [lo] "v"(lo),[hi] "v"(hi),[step] "s"(step),[dst] "v"(dst),',
            '      [alo] "v"(alo),[blo] "v"(blo),[b_lo] "v"(b_lo),[b_hi] "v"(b_hi),',
            '      [b_step] "s"(b_step),[b_dst] "v"(b_dst),' + CONSTANTS,
            '    : "memory","scc","vcc",' + clobbers + ');',
            '}',
            '__device__ __forceinline__ uint2 expand_fp8_for_check(uint32_t source) {',
            '  uint2 output;',
            '  asm volatile(',
        ])
        lines.extend('    "' + op + '\\n\\t"' for op in conversion_ops(
            '%[low]', '%[high]', '%[source]'))
        lines.extend([
            '    : [low] "=&v"(output.x),[high] "=&v"(output.y)',
            '    : [source] "v"(source),' + CONSTANTS + ');',
            '  return output;',
            '}',
        ])
        lines.extend([
            'struct ExpandedFp8 { uint32_t words[8]; };',
            '__device__ __forceinline__ ExpandedFp8 expand_fp8_block_for_check(uint4 source) {',
            '  ExpandedFp8 output;',
            '  asm volatile(',
        ])
        block = [f'v_mov_b32 v{52 + i}, %[in{i}]' for i in range(4)]
        block += block_conversion_ops()
        block += [f'v_mov_b32 %[out{i}], v{52 + i}' for i in range(8)]
        lines.extend('    "' + op + '\\n\\t"' for op in block)
        outputs = ','.join(f'[out{i}] "=&v"(output.words[{i}])' for i in range(8))
        inputs = ','.join(f'[in{i}] "v"(source.{component})'
                          for i, component in enumerate(('x', 'y', 'z', 'w')))
        clobbers = ','.join(f'"v{i}"' for i in range(52, 60))
        lines.extend([
            '    : ' + outputs,
            '    : ' + inputs + ',' + CONSTANTS,
            '    : ' + clobbers + ');',
            '  return output;',
            '}',
        ])
        return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(Fp8Emitter().render())
