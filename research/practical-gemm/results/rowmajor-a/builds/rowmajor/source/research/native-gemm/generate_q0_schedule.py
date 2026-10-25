#!/usr/bin/env python3
"""Two fixed q0 experiments on the late-scale baseline; no parameter sweep.

earlya: move the two future A loads earlier; retain q0 arithmetic split6/18.
balanced: user's13 /15 /15 instruction layout,with pointer adds at the end.
The counts guide scheduling but are not a cycle model for SALU/VMEM/LDS.
"""

import argparse
from pathlib import Path

from generate_fp8 import split_conversion_ops
from generate_scale_latency import ScaleLatencyEmitter


class Q0Emitter(ScaleLatencyEmitter):
    def __init__(self, q0_plan):
        super().__init__('late')
        self.q0_plan = q0_plan

    def emit_stage(self, phase, fetch_future):
        self.emit('s_waitcnt lgkmcnt(0)')
        self.emit('s_barrier')
        self.emit('s_waitcnt vmcnt(0)')
        self.compute_fragment(0, first=0, stop=1)
        self.write_a()
        permutations, arithmetic = split_conversion_ops()
        self.emit_many(permutations)
        if fetch_future:
            self.load_b_packet()
            if self.q0_plan == 'earlya':
                self.advance_b_pointer()
            # Current A payload was consumed by its LDS stores. No later
            # current-stage operation reads v44:51; conversion uses v52:59.
            self.load_a_packets()

        if self.q0_plan == 'earlya':
            # Only A-load positions differ from the late baseline.
            self.emit_many(arithmetic[:6])
            self.compute_fragment(0, first=1, stop=2)
            self.emit_many(arithmetic[6:])
            self.write_b()
            if fetch_future:
                self.advance_source_pointer('%[step]')
            self.compute_fragment(0, first=2)
        else:
            self.compute_fragment(0, first=1, stop=2)
            self.emit_many(arithmetic[:15])
            self.compute_fragment(0, first=2, stop=3)
            self.emit_many(arithmetic[15:])
            self.write_b()
            if fetch_future:
                # Each 64-bit add/carry pair remains adjacent. Bases are
                # updated after their loads and before their next use.
                self.advance_b_pointer()
                self.advance_source_pointer('%[step]')
            self.compute_fragment(0, first=3, stop=4)

        # Identical publication,raw slots and q1/q2/q3 math/scale schedule.
        self.emit('s_waitcnt lgkmcnt(0)')
        self.emit('s_barrier')
        self.read_lds_fragment(0, 0)
        self.read_lds_fragment(3, 4 - phase)
        for fragment in (1, 2):
            self.compute_fragment(fragment)
            self.read_lds_fragment(fragment, fragment)
        self.compute_fragment(3 + phase)

    def render(self):
        text = super().render()
        tag = 1 if self.q0_plan == 'earlya' else 2
        return text.replace('constexpr int kScalePlan=1;',
                            f'constexpr int kScalePlan=1;\nconstexpr int kQ0Plan={tag};')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q0-plan', choices=('earlya', 'balanced'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(Q0Emitter(args.q0_plan).render())
