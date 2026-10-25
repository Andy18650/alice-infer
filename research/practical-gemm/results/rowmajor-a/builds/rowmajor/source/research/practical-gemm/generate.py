#!/usr/bin/env python3
"""Isolate row-major A delivery; retain balanced FP8-B/scaled MFMA pipeline.

K must be a multiple of32 (or zero); no padded row stride. Four consecutive
lanes fetch the four K8 vectors of one row. Each wave produces32 A rows,
not one K8 slice. Scatter b64 stores reconstruct the unchanged consumer LDS.
"""
import argparse
import sys
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1] / 'native-gemm'
sys.path.insert(0, str(NATIVE))
from generate_q0_schedule import Q0Emitter


class RowMajorEmitter(Q0Emitter):
    def __init__(self):
        super().__init__('balanced')
        self.load_index = 0

    @property
    def scalar_register_end(self):
        return 22  # s20:21 = second16-row packet base; s18:19 save EXEC.

    def load_a_packets(self):
        # v40=16*lane. Offset=(lane/4)*lda_bytes+(lane%4)*16.
        # All supported nonzero K have lda_bytes>=64 and <2^24.
        self.emit('v_lshrrev_b32 v43, 6, v40')
        self.emit('v_mad_u32_u24 v43, %[lda_minus64], v43, v40')
        self.emit('s_add_u32 s20, s0, %[a_packet_step]')
        self.emit('s_addc_u32 s21, s1, 0')
        index = self.load_index
        self.load_index += 1
        self.emit('s_cmp_eq_u32 %[a_full], 1')
        self.emit('s_cbranch_scc1 ' + self.label(f'a_full_{index}'))
        # Inactive rows must publish zeros, not stale producer registers.
        self.zero_registers(44, 52)
        for lo, hi, base, limit in ((44, 47, 0, 'a_limit0'), (48, 51, 20, 'a_limit1')):
            self.emit(f'v_cmp_lt_u32 vcc, v43, %[{limit}]')
            self.emit('s_and_saveexec_b64 s[18:19], vcc')
            self.emit(f'global_load_dwordx4 v[{lo}:{hi}], v43, s[{base}:{base + 1}]')
            self.emit('s_mov_b64 exec, s[18:19]')
        self.jump(f'a_loaded_{index}')
        self.mark(f'a_full_{index}')
        self.emit('global_load_dwordx4 v[44:47], v43, s[0:1]')
        self.emit('global_load_dwordx4 v[48:51], v43, s[20:21]')
        self.mark(f'a_loaded_{index}')
        self.restore_b_destination()

    def restore_b_destination(self):
        self.emit('v_add_u32 v43, %[lds_dst], v40')

    def write_a(self):
        # A's old LDS representation puts four K values for two rows into
        # one16-byte record. v52 is dead before the B permutation copies.
        self.emit('v_lshrrev_b32 v52, 2, v40')
        self.emit('v_and_b32 v52, 240, v52')
        self.emit('v_and_b32 v43, 48, v40')
        self.emit('v_lshlrev_b32 v43, 7, v43')
        self.emit('v_or_b32 v43, v43, v52')
        self.emit('v_add_u32 v43, %[a_lds], v43')
        for register, offset in ((44, 0), (46, 512), (48, 256), (50, 768)):
            suffix = f' offset:{offset}' if offset else ''
            self.emit(f'ds_write_b64 v43, v[{register}:{register + 1}]{suffix}')
        self.restore_b_destination()

    def render(self):
        text = super().render()
        old = 'uint32_t rows_left,uint32_t cols_left) {'
        new = ('uint32_t rows_left,uint32_t cols_left,\n'
               ' uint32_t lda_minus64,uint32_t a_packet_step,uint32_t a_lds,\n'
               ' uint32_t a_limit0,uint32_t a_limit1,uint32_t a_full) {')
        assert text.count(old) == 1
        text = text.replace(old, new)
        old = '[rows_left] "s"(rows_left),[cols_left] "s"(cols_left),'
        new = (old + '\n   [lda_minus64] "s"(lda_minus64),[a_packet_step] "s"(a_packet_step),'
               '[a_lds] "s"(a_lds),[a_limit0] "s"(a_limit0),'
               '[a_limit1] "s"(a_limit1),[a_full] "s"(a_full),')
        assert text.count(old) == 1
        return text.replace(old, new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('packed', 'rowmajor'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    emitter = RowMajorEmitter() if args.layout == 'rowmajor' else Q0Emitter('balanced')
    args.output.write_text(emitter.render())
