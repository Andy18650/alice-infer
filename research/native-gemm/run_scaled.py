#!/usr/bin/env python3
"""Matched original/compact/reserved/scaled comparison, with independent refs."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from run import Case, FUNCTIONAL_CASES, capture_smi, run_case, require, write_summary
from run_branch_compare import HERE, ROOT, PROBLEMS
from verify_artifacts import verify_manifest


SOURCES = ('probe.hip','data.hpp','data_fp8.hpp','generate.py','generate_fp8.py',
           'build.sh','build_fp8.sh','inspect_code.py')
SCALED_SOURCES = ('data_scaled.hpp','scaled_kernel.hpp','generate_scaled.py',
                  'build_scaled.sh','inspect_scaled.py')
BUILD_FILES = ('compute.hpp','hashes.txt','inventory.json','loop.txt','body.txt',
               'metadata.txt','compiler-flags.txt')
BOUNDARIES = (95,96,97,127,128,129,191,192,193,255,256,257,
              287,288,289,319,320,321,511,512,513,1023,1024,1025)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(output, records):
    lines = ['# K128/N128 FP32 scale / register-budget comparison','',
             'HIP O3 gfx90a HIP0/BDF31:00.0 CPU+memory NUMA0 block256,automatic clocks,',
             'no model/context/override; full-kernel useful TFLOP/s,packing/reference excluded.',
             'Two9-sample repeats,variant order reversed in repeat2. Scaled uses nonuniform',
             'dyadic FP32 scales; others are unscaled controls,each with its own exact oracle.','',
             '|M x N x K|CTAs|Original144/AGPR64|Compact128|Reserved256|Scaled256|Scaled efficiency %|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for m,n,k in PROBLEMS:
        cells = []
        for variant in ('original','compact','heavy','scaled'):
            rows = sorted((r for r in records if r['variant']==variant and r['kind']=='timing'
                           and (r['m'],r['n'],r['k'])==(m,n,k)),key=lambda r:r['repeat'])
            cells.append('/'.join(f"{r['useful_TFLOPs']:.3f}" for r in rows))
        scaled = sorted((r for r in records if r['variant']=='scaled' and r['kind']=='timing'
                         and (r['m'],r['n'],r['k'])==(m,n,k)),key=lambda r:r['repeat'])
        cells.append('/'.join(f"{100*r['global_clock_adjusted_efficiency']:.3f}" for r in scaled))
        lines.append(f'|{m}x{n}x{k}|{m//128*(n//128)}|'+'|'.join(cells)+'|')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/scaled')
    args=parser.parse_args()
    output=args.output.resolve()
    require(not output.exists() or not any(output.iterdir()),'Choose a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    builds={'original':ROOT/'temp/native-gemm/fp8-schedule/interleaved'}
    builds.update({name:ROOT/f'temp/native-gemm/scaled/{name}' for name in ('compact','heavy','scaled')})
    hashes={}
    for variant,build in builds.items():
        source=HERE/'results/fp8-interleaved/after/source' if variant=='original' else HERE
        verify_manifest(build/'hashes.txt',source)
        hashes[variant]=sha(build/'probe')
        target=output/variant
        for folder in ('source','build','verification','timing','clocks'):
            (target/folder).mkdir(parents=True,exist_ok=True)
        for file in (*SOURCES,*(SCALED_SOURCES if variant!='original' else ())):
            shutil.copy2(source/file,target/'source'/file)
        for file in BUILD_FILES:
            shutil.copy2(build/file,target/'build'/file)
        with (target/'ldd.txt').open('w') as stream:
            subprocess.run(['ldd',str(build/'probe')],stdout=stream,check=True)
    (output/'driver').mkdir()
    for file in ('run_scaled.py','run.py','run_branch_compare.py','verify_artifacts.py'):
        shutil.copy2(HERE/file,output/'driver'/file)
    (output/'invocation.json').write_text(json.dumps(dict(
        builds={key:str(value) for key,value in builds.items()},problems=PROBLEMS,
        scale_layout='FP32 [ceil(K/128)][ceil(N/128)]',scale_pattern=1,
        placement='CPU+memory NUMA0',clock_override=False,
    ),indent=2)+'\n')
    capture_smi(output/'amd-smi-before.txt')
    records=[]
    launches=[]
    with (output/'commands.txt').open('w') as commands:
        def execute(variant,case):
            build=builds[variant]
            require(sha(build/'probe')==hashes[variant],'Executable changed')
            launches.append(dict(variant=variant,case=case.stem,sha256=hashes[variant]))
            commands.write(f'# {variant}\n')
            record=run_case(case,{128:build},output/variant,commands)
            records.append(dict(variant=variant,scale_pattern=case.scale_pattern,
                                b_pattern=case.b_pattern,**record))
            write_summary(output/'summary.csv',records)
            (output/'launch-hashes.json').write_text(json.dumps(launches,indent=2)+'\n')
            if case.timing: report(output,records)

        for variant in ('compact','heavy','scaled'):
            for m,n,k in FUNCTIONAL_CASES:
                execute(variant,Case(128,m,n,k))
            for k in BOUNDARIES:
                execute(variant,Case(128,129,131,k))
            for seed_a,seed_b in ((1730,2718),(1729,2719)):
                execute(variant,Case(128,129,131,257,seed_a=seed_a,seed_b=seed_b))
        # Scale table varies along BOTH K and N; M rows must reuse it. Exact
        # std::fma reference also exercises zero and non-power-of-two scales.
        for pattern in (0,1,2,3):
            for k in (0,1,127,128,129,255,256,257,289,513,1025):
                execute('scaled',Case(128,257,259,k,scale_pattern=pattern))
        for k in (1,7,8,9,31,32,33,63,64):
            execute('scaled',Case(128,129,131,k,scale_pattern=2,b_pattern=1))
        for repeat in (1,2):
            problems=PROBLEMS if repeat==1 else reversed(PROBLEMS)
            order=('original','compact','heavy','scaled')
            if repeat==2: order=tuple(reversed(order))
            for m,n,k in problems:
                for variant in order:
                    execute(variant,Case(128,m,n,k,repeat=repeat))
    capture_smi(output/'amd-smi-after.txt')
    print('COMPLETE',sum(r['samples'] for r in records),'checked launches;',len(launches),'hashed processes')


if __name__=='__main__':
    main()
