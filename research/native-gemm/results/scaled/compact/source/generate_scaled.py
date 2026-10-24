#!/usr/bin/env python3
"""Scalar-address compact W8A16 pipeline and K128 FP32 scale accumulation.

All inline-asm inputs are SGPRs. Lane offsets and four persistent vector
addresses occupy v40:43 instead of vector source-pointer pairs plus external
address inputs. The original five raw slots and20 producer registers stay.
Scaled: v64:127 scaled C, v128:191 raw A, v192:255 raw B (all architected).
Compact control: v64:127 ordinary C; no scaling. Heavy control reserves256
VGPRs but executes exactly the compact-control instruction flow.
"""

import argparse
from pathlib import Path

from generate_fp8 import Fp8Emitter, CONSTANTS


class ScaledEmitter(Fp8Emitter):
    def __init__(self, mode):
        super().__init__('interleaved')
        self.mode = mode
        self.scaled = mode == 'scaled'
        self.stage_index = 0
        self.initial_zero = True

    @property
    def register_end(self):
        return 128 if self.mode == 'compact' else 256

    @property
    def bank(self):
        return (128 + 64 * ((self.stage_index // 4) % 2)) if self.scaled else 64

    def load_a_packets(self):
        for packet in range(2):
            register = 44 + packet * 4
            offset = f' offset:{packet * 1024}' if packet else ''
            self.emit(f'global_load_dwordx4 v[{register}:{register + 3}], v40, s[0:1]{offset}')

    def load_b_packet(self):
        self.emit('global_load_dwordx4 v[60:63], v40, s[2:3]')

    def advance_source_pointer(self, ignored):
        self.emit('s_add_u32 s0, s0, %[a_step]')
        self.emit('s_addc_u32 s1, s1, 0')

    def advance_b_pointer(self):
        self.emit('s_add_u32 s2, s2, %[b_step]')
        self.emit('s_addc_u32 s3, s3, 0')

    def prefetch_global_stage(self):
        self.load_a_packets()
        self.load_b_packet()
        self.advance_source_pointer(None)
        self.advance_b_pointer()

    def write_a(self):
        self.emit('ds_write_b128 v43, v[44:47]')
        self.emit('ds_write_b128 v43, v[48:51] offset:1024')

    def write_b(self):
        self.emit('ds_write_b128 v43, v[52:55] offset:8192')
        self.emit('ds_write_b128 v43, v[56:59] offset:9216')

    def read_lds_fragment(self, fragment, raw_slot):
        for operand, address in ((0, 41), (1, 42)):
            destination = raw_slot * 8 + operand * 4
            offset = f' offset:{fragment * 2048}' if fragment else ''
            self.emit(f'ds_read_b128 v[{destination}:{destination + 3}], v{address}{offset}')

    def scale_microtile(self, previous_bank, microtile, scale):
        for element in range(16):
            index = microtile * 16 + element
            self.emit(f'v_fma_f32 v{64 + index}, v{previous_bank + index}, {scale}, v{64 + index}')

    def compute_fragment(self, raw_slot, first=0, stop=None):
        stop = 4 if stop is None else stop
        for microtile in range(first, stop):
            row, column = divmod(microtile, 2)
            a = raw_slot * 8 + row * 2
            b = raw_slot * 8 + 4 + column * 2
            accumulator = self.bank + microtile * 16
            # Initialize each raw microtile with the first K8 of its K128 group,
            # without64 zero MOVs at every group boundary.
            zero = raw_slot == 0 and (self.stage_index % 4 == 0 if self.scaled else self.initial_zero)
            source = '0' if zero else f'v[{accumulator}:{accumulator + 15}]'
            self.emit(f'v_mfma_f32_32x32x8f16 v[{accumulator}:{accumulator + 15}], '
                      f'v[{a}:{a + 1}], v[{b}:{b + 1}], {source}')
            if self.scaled and self.stage_index % 4 == 0 and raw_slot == 1:
                # Prior group's last q3 write is already separated by the new
                # q0 MFMAs. Drain in microtile completion order while the new
                # q1 MFMA operates on the other64-register raw bank.
                self.scale_microtile(320 - self.bank, microtile, 's15')

    def start_group(self, index):
        self.stage_index = index
        if self.scaled and index % 4 == 0:
            self.emit('s_mov_b32 s15, s14')
            self.emit('s_load_dword s14, s[12:13], 0')
            self.emit('s_add_u32 s12, s12, %[scale_step]')
            self.emit('s_addc_u32 s13, s13, 0')
        if not self.scaled:
            # Only the first ever K8 initializes the single unscaled bank.
            # Separate prefix stage0 below handles it; steady/tails keep C.
            self.initial_zero = False

    def emit_prefix(self):
        for register, argument in ((0, 'a_lo'), (1, 'a_hi'), (2, 'b_lo'), (3, 'b_hi'),
                                   (12, 'scale_lo'), (13, 'scale_hi'), (16, 'tiles')):
            self.emit(f's_mov_b32 s{register}, %[{argument}]')
        self.emit('s_mov_b32 s14, 0')
        self.emit('s_mov_b32 s15, 0')
        self.emit('v_mbcnt_lo_u32_b32 v40, -1, 0')
        self.emit('v_mbcnt_hi_u32_b32 v40, -1, v40')
        self.emit('v_lshlrev_b32 v40, 4, v40')
        self.emit('v_add_u32 v41, %[lds_a], v40')
        self.emit('v_add_u32 v42, %[lds_b], v40')
        self.emit('v_add_u32 v43, %[lds_dst], v40')
        for register in range(64, 128):
            self.emit(f'v_mov_b32 v{register}, 0')
        if self.scaled:
            for register in range(192, 256):
                self.emit(f'v_mov_b32 v{register}, 0')
        self.emit('s_memtime s[4:5]')
        self.emit('s_memrealtime s[8:9]')
        self.emit('s_waitcnt lgkmcnt(0)')
        self.emit('s_cmp_eq_u32 s16, 0')
        self.emit('s_cbranch_scc1 ' + self.label('finish_zero'))
        self.prefetch_global_stage()
        self.emit('s_waitcnt vmcnt(0)')
        self.write_prefetched_stage_to_lds()
        self.emit('s_waitcnt lgkmcnt(0)')
        self.emit('s_barrier')
        self.emit('s_cmp_le_u32 s16, 1')
        self.emit('s_cbranch_scc1 ' + self.label('seed'))
        self.prefetch_global_stage()
        self.mark('seed')
        for fragment in range(4):
            self.read_lds_fragment(fragment, fragment)

    def finish_jump(self, index):
        self.jump('finish_a' if (index // 4) % 2 == 0 else 'finish_b')

    def emit_loop(self):
        # A separate first stage lets the unscaled control use src2=0 once,
        # without a hot per-MFMA initialization condition.
        if not self.scaled:
            self.stage_index = 0
            self.initial_zero = True
            self.emit('s_cmp_eq_u32 s16, 1')
            self.emit('s_cbranch_scc1 ' + self.label('first_last'))
            self.emit('s_cmp_ge_u32 s16, 3')
            self.emit('s_cbranch_scc1 ' + self.label('first_full'))
            self.emit_stage(0, False)
            self.jump('first_complete')
            self.mark('first_full')
            self.emit_stage(0, True)
            self.mark('first_complete')
            self.emit('s_sub_u32 s16, s16, 1')
            self.initial_zero = False
            # First stage used phase0. Align the subsequent8-stage unroll to
            # phase1; no scaling-boundary semantics apply to this control.
            self.jump('control_loop')
            self.mark('first_last')
            self.initial_zero = True
            self.emit_last_stage(0)
            self.finish_jump(0)
            self.initial_zero = False

        self.mark('control_loop' if not self.scaled else 'group_loop')
        self.emit('s_cmp_ge_u32 s16, 10')
        self.emit('s_cbranch_scc1 ' + self.label('steady'))
        self.jump('tail_0')
        self.mark('steady')
        for index in range(8):
            self.start_group(index)
            phase = (index + (0 if self.scaled else 1)) % 2
            self.emit_stage(phase, True)
        self.emit('s_sub_u32 s16, s16, 8')
        self.emit('s_cmp_ge_u32 s16, 10')
        self.emit('s_cbranch_scc1 ' + self.label('steady'))
        for index in range(9):
            self.mark(f'tail_{index}')
            self.start_group(index)
            phase = (index + (0 if self.scaled else 1)) % 2
            self.emit('s_cmp_eq_u32 s16, 1')
            self.emit('s_cbranch_scc1 ' + self.label(f'last_{index}'))
            self.emit('s_cmp_ge_u32 s16, 3')
            self.emit('s_cbranch_scc1 ' + self.label(f'full_{index}'))
            self.emit_stage(phase, False)
            self.jump(f'complete_{index}')
            self.mark(f'full_{index}')
            self.emit_stage(phase, True)
            self.mark(f'complete_{index}')
            self.emit('s_sub_u32 s16, s16, 1')
            self.jump(f'tail_{index + 1}')
            self.mark(f'last_{index}')
            self.emit_last_stage(phase)
            self.finish_jump(index)
        # remaining<=9 here, so tail8 is necessarily final.
        self.mark('tail_9')
        self.finish_jump(8)

    def emit_finish(self):
        for name, bank in (('finish_a', 128), ('finish_b', 192)):
            self.mark(name)
            for _ in range(4):
                self.emit('s_nop 15')
            if self.scaled:
                for microtile in range(4):
                    self.scale_microtile(bank, microtile, 's14')
            self.jump('finish_zero')
        self.mark('finish_zero')
        self.emit('s_nop 3')
        self.emit('s_memtime s[6:7]')
        self.emit('s_memrealtime s[10:11]')
        self.emit('s_waitcnt lgkmcnt(0)')

    def emit_output(self, masked):
        # v3 = per-lane output byte offset from the wave's row/column anchor.
        self.emit('v_lshrrev_b32 v2, 9, v40')
        self.emit('v_mul_lo_u32 v2, %[half_row_bytes], v2')
        self.emit('v_and_b32 v1, 496, v40')
        self.emit('v_lshrrev_b32 v1, 2, v1')
        self.emit('v_add_u32 v3, v1, v2')
        if masked:
            self.emit('v_lshrrev_b32 v5, 4, v40')
            self.emit('v_and_b32 v5, 31, v5')
            self.emit('v_lshrrev_b32 v6, 9, v40')
            self.emit('v_lshlrev_b32 v6, 2, v6')
        for microtile in range(4):
            row, column = divmod(microtile, 2)
            self.emit(f's_mul_i32 s17, %[row_bytes], {row * 32}')
            self.emit(f's_add_u32 s17, s17, {column * 128}')
            self.emit('v_add_u32 v0, s17, v3')
            for element in range(16):
                if masked:
                    delta_row = row * 32 + (element // 4) * 8 + element % 4
                    self.emit(f'v_add_u32 v1, {delta_row}, v6')
                    self.emit(f'v_add_u32 v2, {column * 32}, v5')
                    self.emit('v_cmp_lt_u32 vcc, v1, %[rows_left]')
                    self.emit('s_and_saveexec_b64 s[18:19], vcc')
                    self.emit('v_cmp_lt_u32 vcc, v2, %[cols_left]')
                    self.emit('s_and_b64 exec, exec, vcc')
                register = 64 + microtile * 16 + element
                self.emit(f'global_store_dword v0, v{register}, %[out]')
                if masked:
                    self.emit('s_mov_b64 exec, s[18:19]')
                if element != 15:
                    stride = 'gap_row_bytes' if element % 4 == 3 else 'row_bytes'
                    self.emit(f'v_add_u32 v0, %[{stride}], v0')

    def emit_epilogue(self):
        self.emit('s_cmp_ge_u32 %[rows_left], 64')
        self.emit('s_cbranch_scc0 ' + self.label('masked_output'))
        self.emit('s_cmp_ge_u32 %[cols_left], 64')
        self.emit('s_cbranch_scc0 ' + self.label('masked_output'))
        self.emit_output(False)
        self.jump('output_done')
        self.mark('masked_output')
        self.emit_output(True)
        self.mark('output_done')
        self.emit('v_cmp_eq_u32 vcc, v40, 0')
        self.emit('s_and_saveexec_b64 s[18:19], vcc')
        self.emit('v_mov_b32 v0, 0')
        for offset, register in ((0, 4), (8, 6), (16, 8), (24, 10)):
            self.emit(f'v_mov_b32 v2, s{register}')
            self.emit(f'v_mov_b32 v3, s{register + 1}')
            self.emit(f'global_store_dwordx2 v0, v[2:3], %[stamp] offset:{offset}')
        self.emit('s_mov_b64 exec, s[18:19]')

    def render(self):
        self.emit_prefix()
        self.emit_loop()
        self.emit_finish()
        self.emit_epilogue()
        # Parent emits remaining through a named operand; this kernel owns it.
        self.instructions = [op.replace('%[remaining]', 's16') for op in self.instructions]
        lines = [
            'constexpr int kTile=128,kR=2,kStageK=32,kQ=4,kLdsBytes=16384;',
            f'constexpr bool kResident=false,kScaled={str(self.scaled).lower()};',
            '__device__ __forceinline__ void scaled_pipeline(',
            ' uint32_t a_lo,uint32_t a_hi,uint32_t b_lo,uint32_t b_hi,',
            ' uint32_t a_step,uint32_t b_step,uint32_t lds_a,uint32_t lds_b,uint32_t lds_dst,',
            ' uint32_t tiles,uint32_t scale_lo,uint32_t scale_hi,uint32_t scale_step,',
            ' uint64_t out,uint64_t stamp,uint32_t row_bytes,uint32_t rows_left,uint32_t cols_left) {',
            ' asm volatile(',
        ]
        lines.extend('  "' + op + '\\n\\t"' for op in self.instructions)
        lines += [' :', ' : [a_lo] "s"(a_lo),[a_hi] "s"(a_hi),[b_lo] "s"(b_lo),[b_hi] "s"(b_hi),',
                  '   [a_step] "s"(a_step),[b_step] "s"(b_step),[lds_a] "s"(lds_a),',
                  '   [lds_b] "s"(lds_b),[lds_dst] "s"(lds_dst),[tiles] "s"(tiles),',
                  '   [scale_lo] "s"(scale_lo),[scale_hi] "s"(scale_hi),[scale_step] "s"(scale_step),',
                  '   [out] "s"(out),[stamp] "s"(stamp),[row_bytes] "s"(row_bytes),',
                  '   [half_row_bytes] "s"(row_bytes*4),[gap_row_bytes] "s"(row_bytes*5),',
                  '   [rows_left] "s"(rows_left),[cols_left] "s"(cols_left),' + CONSTANTS]
        clobbers = ['"memory"', '"scc"', '"vcc"']
        clobbers += [f'"s{i}"' for i in range(20)]
        clobbers += [f'"v{i}"' for i in range(self.register_end)]
        lines += [' : ' + ','.join(clobbers) + ');', '}']
        return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('compact', 'heavy', 'scaled'), default='scaled')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(ScaledEmitter(args.mode).render())
