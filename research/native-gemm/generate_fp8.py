#!/usr/bin/env python3
"""Balanced W8A16 producer, with separately auditable scheduling increments.

burst: original in-place conversion and late prefetch.
early: cluster permutations, immediately prefetch B into the freed raw buffer.
interleaved: additionally move current q0/C01 among conversion arithmetic.

Only128x128x32. Packing, consumer reads, matrix order and fences are fixed.
"""

import argparse
from pathlib import Path

from generate import Configuration, PipelineEmitter


CONSTANTS = ('[sel_lo] "s"(0x01040004u),[sel_hi] "s"(0x03040204u),'
             '[mask] "s"(0xbfffbfffu),[scale] "s"(0x5c005c00u)')


def conversion_ops(low, high, source):
    """Four operations/output pair; high first permits original in-place use."""
    result = []
    for destination, selector in ((high, 'sel_hi'), (low, 'sel_lo')):
        result.extend([
            f'v_perm_b32 {destination}, 0, {source}, %[{selector}]',
            f'v_pk_ashrrev_i16 {destination}, 1, {destination} op_sel_hi:[0,1]',
            f'v_and_b32 {destination}, %[mask], {destination}',
            f'v_pk_mul_f16 {destination}, {destination}, %[scale]',
        ])
    return result


def block_conversion_ops(raw_base=52):
    result = []
    for word in reversed(range(4)):
        result.extend(conversion_ops(
            f'v{52 + word * 2}', f'v{53 + word * 2}', f'v{raw_base + word}'))
    return result


def split_conversion_ops():
    """The eight source copies must ALL precede the next raw-buffer load."""
    instructions = block_conversion_ops(raw_base=60)
    permutations = [op for op in instructions if op.startswith('v_perm_')]
    arithmetic = [op for op in instructions if not op.startswith('v_perm_')]
    return permutations, arithmetic


class Fp8Emitter(PipelineEmitter):
    def __init__(self, schedule='interleaved'):
        super().__init__(Configuration(128, 32))
        self.schedule = schedule
        self.initialized_b_pointer = False

    @property
    def raw_base(self):
        return 52 if self.schedule == 'burst' else 60

    @property
    def register_end(self):
        return 60 if self.schedule == 'burst' else 64

    def emit_many(self, instructions):
        for instruction in instructions:
            self.emit(instruction)

    def load_a_packets(self):
        for packet in range(2):
            register = 44 + packet * 4
            displacement = f' offset:{packet * 1024}' if packet else ''
            self.emit(f'global_load_dwordx4 v[{register}:{register + 3}], '
                      f'v[40:41], off{displacement}')

    def load_b_packet(self):
        self.emit(f'global_load_dwordx4 v[{self.raw_base}:{self.raw_base + 3}], '
                  'v[42:43], off')

    def advance_b_pointer(self):
        self.emit('v_add_co_u32 v42, vcc, %[b_step], v42')
        self.emit('v_addc_co_u32 v43, vcc, 0, v43, vcc')

    def prefetch_global_stage(self):
        # Startup only for the new schedules; preserve its original load order.
        if not self.initialized_b_pointer:
            self.emit('v_mov_b32 v42, %[b_lo]')
            self.emit('v_mov_b32 v43, %[b_hi]')
            self.initialized_b_pointer = True
        self.load_a_packets()
        self.load_b_packet()
        self.advance_source_pointer('%[step]')
        self.advance_b_pointer()

    def write_a(self):
        self.emit('ds_write_b128 %[dst], v[44:47]')
        self.emit('ds_write_b128 %[dst], v[48:51] offset:1024')

    def write_b(self):
        self.emit('ds_write_b128 %[b_dst], v[52:55]')
        self.emit('ds_write_b128 %[b_dst], v[56:59] offset:1024')

    def write_prefetched_stage_to_lds(self):
        # Startup has no seeded current operands, hence no MFMA interleaving.
        self.write_a()
        if self.schedule == 'burst':
            self.emit_many(block_conversion_ops())
        else:
            permutations, arithmetic = split_conversion_ops()
            self.emit_many(permutations)
            self.emit_many(arithmetic)
        self.write_b()

    def emit_stage(self, phase, fetch_future):
        if self.schedule == 'burst':
            super().emit_stage(phase, fetch_future)
            return

        # The same retirement/publication fences and raw-slot parity as the
        # measured parent. Only producer instructions and one q0 MFMA move.
        self.emit('s_waitcnt lgkmcnt(0)')
        self.emit('s_barrier')
        self.emit('s_waitcnt vmcnt(0)')
        self.compute_fragment(0, first=0, stop=1)
        self.write_a()
        permutations, arithmetic = split_conversion_ops()
        self.emit_many(permutations)
        if fetch_future:
            # Permutations copied every packed byte out of v60:63. The next
            # load may now complete at any point without changing this stage.
            self.load_b_packet()
            self.advance_b_pointer()

        if self.schedule == 'interleaved':
            # Complete the first two output pairs (six arithmetic VALUs),
            # then start another current MFMA. No fixed64-cycle claim: memory
            # and issue stalls determine the actual elapsed overlap window.
            self.emit_many(arithmetic[:6])
            self.compute_fragment(0, first=1, stop=2)
            self.emit_many(arithmetic[6:])
            first_remaining_mfma = 2
        else:
            self.emit_many(arithmetic)
            first_remaining_mfma = 1

        self.write_b()
        if fetch_future:
            # A load timing stays after all stores, isolating the requested B
            # early-load change rather than adding a second producer retiming.
            self.load_a_packets()
            self.advance_source_pointer('%[step]')
        self.compute_fragment(0, first=first_remaining_mfma)

        self.emit('s_waitcnt lgkmcnt(0)')
        self.emit('s_barrier')
        self.read_lds_fragment(0, 0)
        self.read_lds_fragment(3, 4 - phase)
        for fragment in (1, 2):
            self.compute_fragment(fragment)
            self.read_lds_fragment(fragment, fragment)
        self.compute_fragment(3 + phase)

    def render(self):
        self.emit_startup()
        self.emit_gemm_loop_and_tails()
        self.emit_completion()
        schedule_id = {'burst': 0, 'early': 1, 'interleaved': 2}[self.schedule]
        lines = [
            'constexpr int kTile=128,kR=2,kStageK=32,kQ=4,kLdsBytes=16384;',
            'constexpr bool kResident=false,kFp8BStorage=true;',
            f'constexpr int kFp8Schedule={schedule_id};',
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
        clobbers = ','.join(f'"v{i}"' for i in range(self.register_end))
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
            'struct ExpandedFp8 { uint32_t words[8]; };',
            '__device__ __forceinline__ ExpandedFp8 expand_fp8_block_for_check(uint4 source) {',
            '  ExpandedFp8 output;',
            '  asm volatile(',
        ])
        block = [f'v_mov_b32 v{self.raw_base + i}, %[in{i}]' for i in range(4)]
        if self.schedule == 'burst':
            block += block_conversion_ops()
        else:
            permutations, arithmetic = split_conversion_ops()
            block += permutations
            # Deliberately overwrite the freed source buffer in the bit gate;
            # this must not affect the copied current conversion operands.
            block += [f'v_mov_b32 v{i}, 0x7f7f7f7f' for i in range(60, 64)]
            block += arithmetic
        block += [f'v_mov_b32 %[out{i}], v{52 + i}' for i in range(8)]
        lines.extend('    "' + op + '\\n\\t"' for op in block)
        outputs = ','.join(f'[out{i}] "=&v"(output.words[{i}])' for i in range(8))
        inputs = ','.join(f'[in{i}] "v"(source.{component})'
                          for i, component in enumerate(('x', 'y', 'z', 'w')))
        clobbers = ','.join(f'"v{i}"' for i in range(52, self.register_end))
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
    parser.add_argument('--schedule', choices=('burst', 'early', 'interleaved'), default='interleaved')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(Fp8Emitter(args.schedule).render())
