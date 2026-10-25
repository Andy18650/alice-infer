#!/usr/bin/env python3
"""Matched late-scale baseline,early A loads,and balanced q0 instruction order."""

from run_scaled import HERE, ROOT, PROBLEMS, main


VARIANTS = ('baseline', 'earlya', 'balanced')


def report(output, records):
    lines = [
        '# Early A loads and balanced q0 scheduling', '',
        'HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 block256,automatic clocks.',
        'No model/context/override;full-kernel useful TFLOP/s,packing/H2D/reference excluded.',
        'Two counterbalanced9-sample process medians;all variants use late K128/N128 FP32 scales.', '',
        '|M x N x K|CTAs|Baseline|Early A only|Balanced q0|',
        '|---|---:|---:|---:|---:|',
    ]
    for m, n, k in PROBLEMS:
        columns = []
        for variant in VARIANTS:
            matches = sorted(
                (row for row in records if row['variant'] == variant and row['kind'] == 'timing'
                 and (row['m'], row['n'], row['k']) == (m, n, k)), key=lambda row: row['repeat'],
            )
            columns.append('/'.join(f"{row['useful_TFLOPs']:.3f}" for row in matches))
        lines.append(f'|{m}x{n}x{k}|{m // 128 * (n // 128)}|' + '|'.join(columns) + '|')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(dict(
        builds={
            'baseline': ROOT / 'temp/native-gemm/scale-latency/late',
            **{variant: ROOT / f'temp/native-gemm/q0-schedule/{variant}' for variant in VARIANTS[1:]},
        },
        sources={
            'baseline': HERE / 'results/scale-latency/late/source',
            **{variant: HERE for variant in VARIANTS[1:]},
        },
        output=HERE / 'results/q0-schedule',
        description=__doc__, entry_point='run_q0_schedule.py', report_function=report,
        functional_variants=VARIANTS[1:], rich_variants=VARIANTS[1:],
        extra_sources={
            'baseline': ('generate_scale_latency.py',),
            **{variant: ('generate_scale_latency.py', 'generate_q0_schedule.py') for variant in VARIANTS[1:]},
        },
        experiment_metadata=dict(
            baseline='late scale,six/eighteen conversion split,A loads late',
            earlya='A load instructions moved early,other ordering retained',
            balanced='early A,13/15/15 full-stage q0 gaps,scalar pointer adds after B stores',
        ),
    ))
