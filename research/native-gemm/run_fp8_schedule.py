#!/usr/bin/env python3
"""Two measured FP8 increments: clustered copies/early B, then MFMA interleave."""

import argparse
import sys

from run_branch_compare import HERE, ROOT, main


if __name__ == '__main__':
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--step', choices=('early', 'interleaved'), default='interleaved')
    selected, remaining = selector.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    early = selected.step == 'early'
    before_build = (ROOT / 'temp/native-gemm/fp8-b/s32-t128' if early else
                    ROOT / 'temp/native-gemm/fp8-schedule/early')
    before_source = (HERE / 'results/fp8-b/after/source' if early else
                     HERE / 'results/fp8-early/after/source')
    main(dict(
        before_build=before_build,
        before_source=before_source,
        after_build=ROOT / f'temp/native-gemm/fp8-schedule/{selected.step}',
        after256=None,
        after_source=HERE / f'results/fp8-{selected.step}/after/source',
        output=HERE / f'results/fp8-{selected.step}',
        title=('FP8 clustered permutations + immediate B prefetch' if early else
               'FP8 conversion arithmetic among current MFMAs'),
        description=__doc__,
        operation_change=('Cluster eight copies, B raw buffer v60:63, early B load' if early else
                          'Move current q0/C01 after six conversion arithmetic instructions'),
        entry_point='run_fp8_schedule.py',
        allow_resource_change=early,
        functional_variants=(('before', 128), ('after', 128)),
        extra_sources=('generate_fp8.py', 'data_fp8.hpp', 'build_fp8.sh', 'verify_fp8.hip'),
        before_extra_sources=('generate_fp8.py', 'data_fp8.hpp', 'build_fp8.sh', 'verify_fp8.hip'),
    ))
