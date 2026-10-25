#!/usr/bin/env python3
"""Common-B experiment: unchanged row-major-A/scaled MFMA core, four B formats.

Native retains the previous generated pipeline byte-for-byte. Other formats
load four N-contiguous DWORD rows, transpose4x4 bytes in registers, then reuse
the existing FP8 expansion. Four b64 stores recover the original consumer LDS.
"""
import argparse
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('practical_a',HERE/'generate.py')
practical=importlib.util.module_from_spec(spec);spec.loader.exec_module(practical)
from generate_fp8 import split_conversion_ops

FORMATS=('native','plain','block128','block256')


class CommonBEmitter(practical.RowMajorEmitter):
    def __init__(self,layout):
        super().__init__()
        self.layout=layout
        self.in_prefix=False
        self.prefix_stage=0
        self.loaded_b_stage=0

    def emit_prefix(self):
        self.in_prefix=True
        super().emit_prefix()
        self.in_prefix=False

    def load_b_packet(self):
        self.loaded_b_stage=self.prefix_stage if self.in_prefix else self.stage_index+2
        if self.in_prefix:self.prefix_stage+=1
        # Offset=(lane/32)*4*B_row_stride+(lane%32)*4. v40=lane*16.
        # The factor is16*stride-512, so the result is divisible by4.
        self.emit('v_lshrrev_b32 v43, 9, v40')
        self.emit('v_mad_u32_u24 v43, %[b_offset_factor], v43, v40')
        self.emit('v_lshrrev_b32 v43, 2, v43')
        self.emit('global_load_dword v60, v43, s[2:3]')
        self.emit('s_add_u32 s20, s2, %[b_row_stride]')
        self.emit('s_addc_u32 s21, s3, 0')
        for register in (61,62,63):
            self.emit(f'global_load_dword v{register}, v43, s[20:21]')
            if register!=63:
                self.emit('s_add_u32 s20, s20, %[b_row_stride]')
                self.emit('s_addc_u32 s21, s21, 0')
        self.restore_b_destination()

    def advance_b_pointer(self):
        boundary=self.layout!='plain' and self.loaded_b_stage%4==3
        step='b_group_step' if boundary else 'b_step'
        self.emit(f's_add_u32 s2, s2, %[{step}]')
        self.emit('s_addc_u32 s3, s3, 0')

    def transpose_b(self):
        # First consume every raw word; only then overwrite v60:63. All four
        # intermediates are dead before the usual zero-inserting permutations.
        for dst,a,b,selector in ((52,61,60,'tr_lo'),(53,63,62,'tr_lo'),
                                 (54,61,60,'tr_hi'),(55,63,62,'tr_hi'),
                                 (60,53,52,'tr_out_lo'),(61,53,52,'tr_out_hi'),
                                 (62,55,54,'tr_out_lo'),(63,55,54,'tr_out_hi')):
            self.emit(f'v_perm_b32 v{dst}, v{a}, v{b}, %[{selector}]')

    def emit_many(self,instructions):
        if instructions==split_conversion_ops()[0]:self.transpose_b()
        super().emit_many(instructions)

    def write_b(self):
        # lane bits: l[0:2]->LDS[6:8], l3->3, l4->10, l5->9.
        # v41 is temporarily dead: restore the A consumer address before reads.
        self.emit('v_lshlrev_b32 v43, 2, v40')
        self.emit('v_and_b32 v43, 448, v43')
        self.emit('v_lshrrev_b32 v41, 4, v40')
        self.emit('v_and_b32 v41, 8, v41')
        self.emit('v_or_b32 v43, v43, v41')
        self.emit('v_lshlrev_b32 v41, 2, v40')
        self.emit('v_and_b32 v41, 1024, v41')
        self.emit('v_or_b32 v43, v43, v41')
        self.emit('v_and_b32 v41, 512, v40')
        self.emit('v_or_b32 v43, v43, v41')
        self.emit('v_add_u32 v43, %[lds_dst], v43')
        for register,offset in ((52,8192),(54,8208),(56,8224),(58,8240)):
            self.emit(f'ds_write_b64 v43, v[{register}:{register+1}] offset:{offset}')
        self.emit('v_add_u32 v41, %[lds_a], v40')
        self.restore_b_destination()

    def render(self):
        text=super().render()
        old='uint32_t a_limit0,uint32_t a_limit1,uint32_t a_full) {'
        new=('uint32_t a_limit0,uint32_t a_limit1,uint32_t a_full,\n'
             ' uint32_t b_row_stride,uint32_t b_offset_factor,uint32_t b_group_step) {')
        assert text.count(old)==1;text=text.replace(old,new)
        old='[a_limit1] "s"(a_limit1),[a_full] "s"(a_full),'
        new=(old+'\n   [b_row_stride] "s"(b_row_stride),[b_offset_factor] "s"(b_offset_factor),'
             '[b_group_step] "s"(b_group_step),\n'
             '   [tr_lo] "s"(0x05010400u),[tr_hi] "s"(0x07030602u),'
             '[tr_out_lo] "s"(0x05040100u),[tr_out_hi] "s"(0x07060302u),')
        assert text.count(old)==1;return text.replace(old,new)


def kernel_text(layout):
    text=(HERE/'rowmajor_kernel.hpp').read_text()
    if layout=='native':return text
    replacements={
        'const uintptr_t b_base = reinterpret_cast<uintptr_t>(b) + size_t(tile_column) * 4096 + wave * 1024;':
        '''const uint32_t bn = COMMON_B_FORMAT == 2 ? 128 : 256;
    const uint32_t b_stride = COMMON_B_FORMAT == 1 ? uint32_t(np) : bn;
    const uint32_t storage_n = COMMON_B_FORMAT == 1 ? uint32_t(np) : (uint32_t(np)+bn-1)/bn*bn;
    const size_t b_panel = COMMON_B_FORMAT == 1 ? size_t(tile_column)*128
        : size_t(tile_column/(bn/128))*128*bn+(tile_column%(bn/128))*128;
    const uintptr_t b_base = reinterpret_cast<uintptr_t>(b)+b_panel+wave*8*b_stride;''',
        '__builtin_amdgcn_readfirstlane(uint32_t(np * 32)),':
        '__builtin_amdgcn_readfirstlane(uint32_t(b_stride * 32)),',
        '__builtin_amdgcn_readfirstlane(uint32_t(a_rows == 32)));':
        '''__builtin_amdgcn_readfirstlane(uint32_t(a_rows == 32)),
        __builtin_amdgcn_readfirstlane(b_stride),
        __builtin_amdgcn_readfirstlane(uint32_t(16*b_stride-512)),
        __builtin_amdgcn_readfirstlane(uint32_t(128*storage_n-96*bn)));''',
    }
    for old,new in replacements.items():
        assert text.count(old)==1,old;text=text.replace(old,new)
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format',choices=FORMATS,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    emitter=practical.RowMajorEmitter() if a.format=='native' else CommonBEmitter(a.format)
    a.output.write_text(emitter.render())
    a.output.with_name('kernel.hpp').write_text(kernel_text(a.format))
