#!/usr/bin/env python3
"""Generate the fixed-register gfx90a GEMM pipeline.

The ordinary GEMM loop keeps its peeled control flow. One existing current-q0
MFMA is issued immediately after the head VM0, before next-panel LDS stores.
All memory order, per-accumulator MFMA order and register allocation are unchanged.
Resident control is unchanged.

Normal GEMM: one LDS buffer, global prefetch, rolling operand reads.
Resident control: seed one StageK fragment and repeat it, without loop memory
traffic. It is a diagnostic, not ordinary GEMM.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Configuration:
    tile: int
    stage_k: int
    resident: bool = False

    @property
    def micro_tiles_per_wave_side(self):
        # Four waves form a 2x2 grid; each microtile is 32x32.
        return self.tile // 64

    @property
    def micro_pairs_per_operand(self):
        # One b128 transfer holds two adjacent MFMA operand fragments.
        return self.micro_tiles_per_wave_side // 2

    @property
    def fragments_per_stage(self):
        return self.stage_k // 8

    @property
    def registers_per_raw_slot(self):
        # Each microfragment uses two VGPRs, for each of A and B.
        return 4 * self.micro_tiles_per_wave_side

    @property
    def global_pointer_register(self):
        # Q current fragments plus one spare last-fragment slot.
        return (self.fragments_per_stage + 1) * self.registers_per_raw_slot

    @property
    def producer_register_base(self):
        # Leave room for the 64-bit source pointer; align vector tuples to four.
        return (self.global_pointer_register + 2 + 3) // 4 * 4

    @property
    def global_packets_per_stage(self):
        return self.fragments_per_stage * self.micro_pairs_per_operand

    @property
    def fixed_register_end(self):
        return self.producer_register_base + self.global_packets_per_stage * 4


class PipelineEmitter:
    def __init__(self, config):
        self.config = config
        self.instructions = []

    def emit(self, instruction):
        self.instructions.append(instruction)

    def label(self, name):
        # %= makes inline-asm labels unique after inlining.
        return f".Lnative_{self.config.tile}_{self.config.stage_k}_{name}_%="

    def mark(self, name):
        self.emit(self.label(name) + ":")

    def jump(self, name):
        self.emit("s_branch " + self.label(name))

    def advance_source_pointer(self, increment):
        register = self.config.global_pointer_register
        self.emit(f"v_add_co_u32 v{register}, vcc, {increment}, v{register}")
        self.emit(
            f"v_addc_co_u32 v{register + 1}, vcc, 0, v{register + 1}, vcc"
        )

    def prefetch_global_stage(self):
        """Load the next panel into producer VGPRs and advance to its successor.

        A DWORDx4 instruction transfers 16 bytes/lane, or 1024 bytes/wave.
        Immediate offsets are limited to one 4 KiB group here. Larger panels
        advance the pointer between groups. C++ supplies %[step] with those
        internal increments subtracted from the full K-stage source stride.
        """
        config = self.config
        pointer = config.global_pointer_register

        for packet in range(config.global_packets_per_stage):
            if packet and packet % 4 == 0:
                self.advance_source_pointer(4096)

            offset = (packet % 4) * 1024
            destination = config.producer_register_base + 4 * packet
            displacement = f" offset:{offset}" if offset else ""
            self.emit(
                f"global_load_dwordx4 v[{destination}:{destination + 3}], "
                f"v[{pointer}:{pointer + 1}], off{displacement}"
            )

        self.advance_source_pointer("%[step]")

    def write_prefetched_stage_to_lds(self):
        """Each wave writes its own A or B half-panel; no inter-wave overlap."""
        config = self.config
        pairs = config.micro_pairs_per_operand

        for fragment in range(config.fragments_per_stage):
            for pair in range(pairs):
                offset = fragment * config.tile * 16 + pair * 1024
                source = config.producer_register_base + 4 * (fragment * pairs + pair)
                displacement = f" offset:{offset}" if offset else ""
                self.emit(
                    f"ds_write_b128 %[dst], v[{source}:{source + 3}]{displacement}"
                )

    def read_lds_fragment(self, fragment, raw_slot):
        """Read both operands of a K8 slice into a fixed raw register slot."""
        config = self.config
        micro_tiles = config.micro_tiles_per_wave_side

        for operand_index, address in ((0, "alo"), (1, "blo")):
            for pair in range(config.micro_pairs_per_operand):
                offset = fragment * config.tile * 16 + pair * 1024
                destination = (
                    raw_slot * config.registers_per_raw_slot
                    + operand_index * 2 * micro_tiles
                    + pair * 4
                )
                displacement = f" offset:{offset}" if offset else ""
                self.emit(
                    f"ds_read_b128 v[{destination}:{destination + 3}], "
                    f"%[{address}]{displacement}"
                )

    def compute_fragment(self, raw_slot, first=0, stop=None):
        """Outer product of the wave's A and B microfragments.

        Selected 128 tile: A0/A1 and B0/B1 update C00/C01/C10/C11.
        Each input microfragment is used twice, without another LDS read.
        """
        config = self.config
        micro_tiles = config.micro_tiles_per_wave_side
        base = raw_slot * config.registers_per_raw_slot

        end = micro_tiles * micro_tiles if stop is None else stop
        for accumulator in range(first, end):
            row, column = divmod(accumulator, micro_tiles)
            a_register = base + 2 * row
            b_register = base + 2 * micro_tiles + 2 * column
            self.emit(
                f"v_mfma_f32_32x32x8f16 %[c{accumulator}], "
                f"v[{a_register}:{a_register + 1}], "
                f"v[{b_register}:{b_register + 1}], %[c{accumulator}]"
            )

    def emit_startup(self):
        self.emit("s_memtime %[begin]")
        self.emit("s_memrealtime %[wall_begin]")
        self.emit("s_waitcnt lgkmcnt(0)")
        self.emit("s_cmp_eq_u32 %[remaining], 0")
        self.emit("s_cbranch_scc1 " + self.label("done"))

        pointer = self.config.global_pointer_register
        self.emit(f"v_mov_b32 v{pointer}, %[lo]")
        self.emit(f"v_mov_b32 v{pointer + 1}, %[hi]")

        self.prefetch_global_stage()
        self.emit("s_waitcnt vmcnt(0)")
        self.write_prefetched_stage_to_lds()
        self.emit("s_waitcnt lgkmcnt(0)")
        self.emit("s_barrier")  # Publish initial stage.

        self.emit("s_cmp_le_u32 %[remaining], 1")
        self.emit("s_cbranch_scc1 " + self.label("seed"))
        self.prefetch_global_stage()

        self.mark("seed")
        for fragment in range(self.config.fragments_per_stage):
            self.read_lds_fragment(fragment, fragment)

    def emit_stage(self, phase, fetch_future):
        """One non-final stage, with statically known future-load validity.

        This emits no loop-control instructions and does not update remaining.
        Its memory/wait/barrier/MFMA sequence is the original stage sequence.
        """
        config = self.config
        fragments = config.fragments_per_stage
        current_last_slot = fragments - 1 + phase
        spare_last_slot = fragments - phase

        # RETIRE: every wave must finish reading the current LDS contents
        # before any wave overwrites the shared buffer with the next stage.
        self.emit("s_waitcnt lgkmcnt(0)")
        self.emit("s_barrier")
        self.emit("s_waitcnt vmcnt(0)")
        # Start one ready current-stage matrix operation before staging work.
        # It is independent of producer VGPRs/LDS; do not duplicate it below.
        self.compute_fragment(0, first=0, stop=1)
        self.write_prefetched_stage_to_lds()

        # The register producer buffer is free after the LDS writes issue.
        # The caller, not a hot-path branch, establishes whether stage+2 exists.
        if fetch_future:
            self.prefetch_global_stage()

        publish_after = fragments // 4
        for fragment in range(publish_after):
            self.compute_fragment(fragment, first=1 if fragment == 0 else 0)

        # PUBLISH: finish all next-stage LDS writes before any next-stage read.
        # This does not drain the future global prefetch (VMEM counter).
        self.emit("s_waitcnt lgkmcnt(0)")
        self.emit("s_barrier")
        for fragment in range(publish_after):
            self.read_lds_fragment(fragment, fragment)
        self.read_lds_fragment(fragments - 1, spare_last_slot)

        # Reuse a raw slot only after issuing all current math that consumes it.
        for fragment in range(publish_after, fragments - 1):
            self.compute_fragment(fragment)
            self.read_lds_fragment(fragment, fragment)
        self.compute_fragment(current_last_slot)

    def emit_last_stage(self, phase):
        """Consume the final already-prefetched operands, without any future IO."""
        self.emit("s_waitcnt lgkmcnt(0)")
        for fragment in range(self.config.fragments_per_stage - 1):
            self.compute_fragment(fragment)
        self.compute_fragment(self.config.fragments_per_stage - 1 + phase)

    def emit_gemm_loop_and_tails(self):
        # Both stages of an unrolled pair may fetch stage+2 only when at least
        # four stages remain at pair entry. A pair restores q3's phase0 parity.
        self.emit("s_cmp_lt_u32 %[remaining], 4")
        self.emit("s_cbranch_scc1 " + self.label("tail_dispatch"))
        self.mark("steady")
        self.emit_stage(phase=0, fetch_future=True)
        self.emit_stage(phase=1, fetch_future=True)
        self.emit("s_sub_u32 %[remaining], %[remaining], 2")
        self.emit("s_cmp_ge_u32 %[remaining], 4")
        self.emit("s_cbranch_scc1 " + self.label("steady"))

        # Startup handles zero stages. Here one, two or three remain, always
        # with q3 in the phase0 slot. Tails are statically bounded and no longer
        # need to modify the local remaining counter (unused after the asm).
        self.mark("tail_dispatch")
        self.emit("s_cmp_eq_u32 %[remaining], 1")
        self.emit("s_cbranch_scc1 " + self.label("tail_one"))
        self.emit("s_cmp_eq_u32 %[remaining], 2")
        self.emit("s_cbranch_scc1 " + self.label("tail_two"))

        self.mark("tail_three")
        self.emit_stage(phase=0, fetch_future=True)
        self.emit_stage(phase=1, fetch_future=False)
        self.emit_last_stage(phase=0)
        self.jump("done")

        self.mark("tail_two")
        self.emit_stage(phase=0, fetch_future=False)
        self.emit_last_stage(phase=1)
        self.jump("done")

        self.mark("tail_one")
        self.emit_last_stage(phase=0)
        # Fall through to done; do not emit an s_branch 0.

    def emit_resident_control(self):
        # Deliberately not normal GEMM: repeat the first seeded panel.
        self.emit("s_waitcnt lgkmcnt(0)")
        self.mark("resident")
        for fragment in range(self.config.fragments_per_stage):
            self.compute_fragment(fragment)
        self.emit("s_sub_u32 %[remaining], %[remaining], 1")
        self.emit("s_cmp_lg_u32 %[remaining], 0")
        self.emit("s_cbranch_scc1 " + self.label("resident"))

    def emit_completion(self):
        self.mark("done")
        # Conservative constant matrix-tail gap, never in the hot backedge.
        for _ in range(4):
            self.emit("s_nop 15")
        self.emit("s_memtime %[end]")
        self.emit("s_memrealtime %[wall_end]")
        self.emit("s_waitcnt lgkmcnt(0)")

    def render(self):
        """Render the unchanged ABI wrapper around the selected instruction flow."""
        config = self.config
        tile = config.tile
        stage = config.stage_k
        micro_tiles = config.micro_tiles_per_wave_side
        fragments = config.fragments_per_stage

        self.emit_startup()
        if config.resident:
            self.emit_resident_control()
        else:
            self.emit_gemm_loop_and_tails()
        self.emit_completion()

        lines = [
            f'constexpr int kTile={tile},kR={micro_tiles},kStageK={stage},kQ={fragments},kLdsBytes={tile * stage * 4};',
            f'constexpr bool kResident={str(config.resident).lower()};',
            '__device__ __forceinline__ void pipeline(uint32_t lo,uint32_t hi,uint32_t step,uint32_t dst,uint32_t alo,uint32_t blo,int tiles,F16 (&c)[kR*kR],uint64_t& begin,uint64_t& end,uint64_t& wall_begin,uint64_t& wall_end) {',
            '  uint32_t remaining=tiles;',
            '  asm volatile(',
        ]
        lines.extend('    "' + op + '\\n\\t"' for op in self.instructions)
        accumulators = ',\n      '.join(
            f'[c{i}] "+a"(c[{i}])' for i in range(micro_tiles * micro_tiles)
        )
        clobbers = ','.join(f'"v{i}"' for i in range(config.fixed_register_end))
        lines.extend([
            '    : ' + accumulators + ',',
            '      [remaining] "+&s"(remaining),[begin] "=&s"(begin),[end] "=&s"(end),[wall_begin] "=&s"(wall_begin),[wall_end] "=&s"(wall_end)',
            '    : [lo] "v"(lo),[hi] "v"(hi),[step] "s"(step),[dst] "v"(dst),[alo] "v"(alo),[blo] "v"(blo)',
            '    : "memory","scc","vcc",' + clobbers + ');',
            '}',
        ])
        return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tile', type=int, choices=(128, 256), default=128)
    parser.add_argument('--stage-k', type=int, choices=(32, 64), default=32)
    parser.add_argument('--resident', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    config = Configuration(args.tile, args.stage_k, args.resident)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(PipelineEmitter(config).render())


if __name__ == '__main__':
    main()
