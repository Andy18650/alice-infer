#!/usr/bin/env python3
"""Matched scalar / packed-move / packed-move+FMA scaled GEMM comparison."""

from run_scaled import HERE, ROOT, PROBLEMS, main


VARIANTS = ('scalar', 'mov', 'packed')


def report(output, records):
    lines = [
        '# Packed FP32 scaled GEMM', '',
        'Native HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 block256, automatic clocks.',
        'No model/context/clock override. Full-kernel useful TFLOP/s; packing/H2D/reference excluded.',
        'Two counterbalanced nine-sample process medians; nonuniform K128/N128 FP32 scales.', '',
        '|M x N x K|CTAs|Scalar|Packed moves|Packed moves+FMA|Final efficiency %|',
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
        final = sorted(
            (row for row in records if row['variant'] == 'packed'
             and row['kind'] == 'timing' and (row['m'], row['n'], row['k']) == (m, n, k)),
            key=lambda row: row['repeat'],
        )
        columns.append('/'.join(f"{100 * row['global_clock_adjusted_efficiency']:.3f}" for row in final))
        lines.append(f'|{m}x{n}x{k}|{m // 128 * (n // 128)}|' + '|'.join(columns) + '|')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(dict(
        builds={
            'scalar': ROOT / 'temp/native-gemm/scaled/scaled',
            'mov': ROOT / 'temp/native-gemm/scaled-packed-fixed/scaled-mov',
            'packed': ROOT / 'temp/native-gemm/scaled-packed-fixed/scaled-all',
        },
        sources={
            'scalar': HERE / 'results/scaled/scaled/source',
            'mov': HERE / 'results/scaled-packed-fixed/mov/source',
            'packed': HERE / 'results/scaled-packed-fixed/packed/source',
        },
        output=HERE / 'results/scaled-packed-fixed',
        description=__doc__,
        entry_point='run_scaled_packed.py',
        report_function=report,
        functional_variants=('mov', 'packed'),
        rich_variants=('mov', 'packed'),
    ))
