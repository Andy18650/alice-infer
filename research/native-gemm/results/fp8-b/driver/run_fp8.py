#!/usr/bin/env python3
"""Balanced all-wave W8A16 versus preserved FP16 MFMA-first kernel."""

from run_branch_compare import HERE, ROOT, main


if __name__ == '__main__':
    main(dict(
        before_build=ROOT / 'temp/native-gemm/mfma-first/s32-t128',
        before_source=HERE / 'results/mfma-first/after/source',
        after_build=ROOT / 'temp/native-gemm/fp8-b/s32-t128',
        after256=None,
        after_source=HERE,
        output=HERE / 'results/fp8-b',
        title='Balanced all-wave FP8 B versus FP16 B',
        description=__doc__,
        operation_change='FP8 B, all waves produce one K8 A/B slice, A/B repacked, B expanded before LDS',
        entry_point='run_fp8.py',
        allow_resource_change=True,
        functional_variants=(('before', 128), ('after', 128)),
        extra_sources=('generate_fp8.py', 'data_fp8.hpp', 'build_fp8.sh', 'verify_fp8.hip'),
    ))
