#!/usr/bin/env python3
"""Three bounded experiments, not an auto-tuner.

late: load current scale after q1 drains the previous raw bank.
batch: load a pair of K scales there, using N-major padded metadata/SGPR queue.
read: late + issue next q1 raw reads before the last packed microtile drain.
"""

import argparse
from pathlib import Path

from generate_scaled import ScaledEmitter


class ScaleLatencyEmitter(ScaledEmitter):
    def __init__(self, plan):
        super().__init__('scaled', 'all')
        self.plan = plan
        self.in_last_stage = False
        self.skip_q1_read = False

    @property
    def scalar_register_end(self):
        return 22 if self.plan == 'batch' else 20

    def start_group(self, index):
        self.stage_index = index
        if index % 4 == 0:
            # Previous current-scale load was completed by an intervening
            # existing LGKM0. No new load is issued before this head wait.
            self.emit('s_mov_b32 s15, s14')
            if self.plan == 'batch' and index % 8 == 4:
                # Queue high belongs to the second K128 group. Queue was
                # established by group A's second-stage retirement LGKM0.
                self.emit('s_mov_b32 s14, s21')

    def issue_scale_load(self):
        if self.plan == 'batch':
            if self.stage_index % 8 != 0:
                return
            self.emit('s_load_dwordx2 s[20:21], s[12:13], 0')
            self.emit('s_add_u32 s12, s12, 8')
        else:
            self.emit('s_load_dword s14, s[12:13], 0')
            self.emit('s_add_u32 s12, s12, %[scale_step]')
        self.emit('s_addc_u32 s13, s13, 0')

    def compute_fragment(self, raw_slot, first=0, stop=None):
        super().compute_fragment(raw_slot, first, stop)
        if self.plan == 'batch' and self.stage_index % 8 == 1 and raw_slot == 0 and first == 0:
            # This q0 follows the existing head LGKM0, so queue low is ready.
            # Copy after the MFMA rather than delaying its immediate issue.
            self.emit('s_mov_b32 s14, s20')
        if self.stage_index % 4 == 0 and raw_slot == 1:
            # All previous-scale FMAs are issued. Current scale is not used
            # until a later group handoff or final flush. q2/q3 now hide SMEM
            # before the next stage's existing retirement LGKM0.
            self.issue_scale_load()

    def scale_microtile(self, previous_bank, microtile, scale):
        if (self.plan == 'read' and not self.in_last_stage
                and scale == 's15' and microtile == 3):
            # This reproduces the pre-scaling pipeline's immediate q1 operand
            # slot refill after its last MFMA. These are SrcA/B VGPRs, not
            # MFMA destination/SrcC registers. Never do this in a final stage.
            super().read_lds_fragment(1, 1)
            self.skip_q1_read = True
        super().scale_microtile(previous_bank, microtile, scale)

    def read_lds_fragment(self, fragment, raw_slot):
        if fragment == 1 and raw_slot == 1 and self.skip_q1_read:
            self.skip_q1_read = False
            return
        super().read_lds_fragment(fragment, raw_slot)

    def emit_last_stage(self, phase):
        self.in_last_stage = True
        super().emit_last_stage(phase)
        self.in_last_stage = False

    def emit_finish(self):
        for name, bank in (('finish_a', 128), ('finish_b', 192)):
            self.mark(name)
            # Required for a single-stage last group: its deferred scalar load
            # has no intervening next-stage head wait before final consumption.
            self.emit('s_waitcnt lgkmcnt(0)')
            if self.plan == 'batch' and bank == 128:
                self.emit('s_mov_b32 s14, s20')
            for _ in range(4):
                self.emit('s_nop 15')
            for microtile in range(4):
                self.scale_microtile(bank, microtile, 's14')
            self.jump('finish_zero')
        self.mark('finish_zero')
        self.emit('s_nop 3')
        self.emit('s_memtime s[6:7]')
        self.emit('s_memrealtime s[10:11]')
        self.emit('s_waitcnt lgkmcnt(0)')

    def render(self):
        text = super().render()
        tag = {'late': 1, 'batch': 2, 'read': 3}[self.plan]
        return text.replace('constexpr bool kResident=false,kScaled=true;',
                            f'constexpr bool kResident=false,kScaled=true;\nconstexpr int kScalePlan={tag};')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', choices=('late', 'batch', 'read'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(ScaleLatencyEmitter(args.plan).render())
