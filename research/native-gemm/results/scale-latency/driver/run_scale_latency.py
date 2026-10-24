#!/usr/bin/env python3
"""Four-way baseline/late-scale/pair-queue/late+early-q1-read comparison."""

from run_scaled import HERE, ROOT, PROBLEMS, main


VARIANTS = ('baseline', 'late', 'batch', 'read')


def report(output, records):
    lines = [
        '# Scale metadata and packed-drain scheduling', '',
        'Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 block256,automatic clocks.',
        'No model/context/clock override. Full-kernel useful TFLOP/s;packing/H2D/reference excluded.',
        'Two counterbalanced9-sample process medians. Read variant is based on late,not batch.', '',
        '|M x N x K|CTAs|Baseline|Late scale|Batched pair|Late+early q1 read|',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for m, n, k in PROBLEMS:
        columns = []
        for variant in VARIANTS:
            rows = sorted(
                (row for row in records if row['variant'] == variant
                 and row['kind'] == 'timing' and (row['m'], row['n'], row['k']) == (m, n, k)),
                key=lambda row: row['repeat'],
            )
            columns.append('/'.join(f"{row['useful_TFLOPs']:.3f}" for row in rows))
        lines.append(f'|{m}x{n}x{k}|{m // 128 * (n // 128)}|' + '|'.join(columns) + '|')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(dict(
        builds={
            'baseline': ROOT / 'temp/native-gemm/scaled-packed-fixed/scaled-all',
            **{variant: ROOT / f'temp/native-gemm/scale-latency/{variant}'
               for variant in VARIANTS[1:]},
        },
        sources={
            'baseline': HERE / 'results/scaled-packed-fixed/packed/source',
            **{variant: HERE for variant in VARIANTS[1:]},
        },
        output=HERE / 'results/scale-latency',
        description=__doc__,
        entry_point='run_scale_latency.py',
        report_function=report,
        functional_variants=VARIANTS[1:],
        rich_variants=VARIANTS[1:],
        extra_sources={variant: ('generate_scale_latency.py',) for variant in VARIANTS[1:]},
        experiment_metadata=dict(
            baseline='K-major single scale at group head',
            late='K-major single scale after previous-bank q1 drain',
            batch='N-major K-pairs padded to even,SGPRs20:21;deferred low/high consumption',
            read='late plus q1 operand refill before last packed microtile drain',
        ),
    ))
