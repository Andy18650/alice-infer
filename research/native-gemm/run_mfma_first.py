#!/usr/bin/env python3
"""Move one current-q0 MFMA after head VM0; same104/208/312-grid A/B protocol."""

from run_branch_compare import HERE, ROOT, main


if __name__ == '__main__':
    main(dict(
        before_build=ROOT / 'temp/native-gemm/branch-clean/s32-t128',
        before_source=HERE / 'results/branch-clean/after/source',
        after_build=ROOT / 'temp/native-gemm/mfma-first/s32-t128',
        after256=ROOT / 'temp/native-gemm/mfma-first/s32-t256',
        after_source=HERE,
        output=HERE / 'results/mfma-first',
        title='One MFMA immediately after head VM0',
        description=__doc__,
        operation_change='Move existing first current-q0 MFMA after head VM0 before LDS writes',
        entry_point='run_mfma_first.py',
    ))
