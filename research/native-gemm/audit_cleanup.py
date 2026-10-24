#!/usr/bin/env python3
"""Prove the readability cleanup preserves packing and native computation.

Checks all eight generator configurations, complete CPU-packed vectors over
boundary/seed cases, selected native .text bytes, and the bank-control .text.
Historical results are read only. Default after-source/build paths pin the
readability-cleanup snapshot, not later intentional algorithm changes. New
compiler outputs go to --work and the machine-readable proof to --output.
"""

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gpu_text(code_object):
    sections = json.loads(subprocess.check_output([
        'llvm-readobj-21', '--sections', '--elf-output-style=JSON', str(code_object),
    ]))[0]['Sections']
    section = next(item['Section'] for item in sections
                   if item['Section']['Name']['Name'] == '.text')
    image = code_object.read_bytes()
    return image[section['Offset']:section['Offset'] + section['Size']]


def check_generators(baseline, after_source, work):
    proofs = []
    for tile, stage_k, resident in itertools.product((128, 256), (32, 64), (False, True)):
        name = f't{tile}-s{stage_k}-r{int(resident)}'
        extra = ['--resident'] if resident else []
        files = []
        for label, script in (('before', baseline / 'source/generate.py'),
                              ('after', after_source / 'generate.py')):
            output = work / f'{name}-{label}.hpp'
            subprocess.run([
                'python3', str(script), '--tile', str(tile), '--stage-k', str(stage_k),
                *extra, '--output', str(output),
            ], check=True)
            files.append(output)
        before, after = (file.read_bytes() for file in files)
        require(before == after, f'Generated header changed: {name}')
        proofs.append(dict(configuration=name,bytes=len(after),sha256=digest(after)))
    return proofs


def packing_test_source(old_header, new_header):
    return '''#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "NEW_HEADER"
namespace baseline {
#include "OLD_HEADER"
}

int main() {
    struct Shape { int m, n, k; };
    const Shape shapes[] = {
        {1, 1, 0}, {3, 7, 1}, {17, 23, 7}, {31, 35, 8},
        {129, 131, 33}, {129, 131, 65}, {255, 257, 127},
        {257, 259, 129}, {1, 3, 65535},
    };
    size_t packed_vectors = 0;
    size_t matrix_pairs = 0;
    for (const auto shape : shapes) {
        for (uint32_t changed_seed : {0u, 1u}) {
            baseline::Data old(shape.m, shape.n, shape.k,
                               1729 + changed_seed, 2718 + changed_seed);
            TestMatrices current(shape.m, shape.n, shape.k,
                                 1729 + changed_seed, 2718 + changed_seed);
            if (old.a != current.row_major_a_units || old.b != current.row_major_b_units)
                throw std::runtime_error("logical matrix generation changed");
            for (int unit = -4; unit <= 4; ++unit)
                if (baseline::half_bits(unit) != half_bits_from_unit(unit))
                    throw std::runtime_error("FP16 encoding changed");
            for (int tile : {128, 256}) {
                for (int stage : {32, 64}) {
                    for (bool operand_b : {false, true}) {
                        if (old.pack(operand_b, tile, stage)
                            != current.pack_operand(operand_b, tile, stage))
                            throw std::runtime_error("packed input bits changed");
                        ++packed_vectors;
                    }
                    for (int row : {0, shape.m - 1}) {
                        for (int column : {0, shape.n - 1}) {
                            if (old.ref(row, column) != current.reference_dot(row, column)
                                || old.ref(row, column, std::min(shape.k, stage))
                                   != current.reference_dot(row, column, std::min(shape.k, stage)))
                                throw std::runtime_error("row-major reference changed");
                        }
                    }
                }
            }
            ++matrix_pairs;
        }
    }
    std::printf("matrix_pairs=%zu packed_vectors=%zu bitwise_equal=1\\n",
                matrix_pairs, packed_vectors);
}
'''.replace('OLD_HEADER', str(old_header)).replace('NEW_HEADER', str(new_header))


def check_packing(baseline, after_source, work, output):
    source = work / 'packing-test.cpp'
    source.write_text(packing_test_source(baseline / 'source/data.hpp', after_source / 'data.hpp'))
    program = work / 'packing-test'
    command = ['c++', '-O2', '-std=c++20', str(source), '-o', str(program)]
    subprocess.run(command, check=True)
    result = subprocess.check_output([str(program)], text=True)
    shutil.copy2(source, output / source.name)
    (output / 'packing-test.txt').write_text(result)
    return dict(command=command,source_sha256=digest(source.read_bytes()),result=result.strip())


def compare_code_objects(before, after, description):
    old_text = gpu_text(before)
    new_text = gpu_text(after)
    require(old_text == new_text, f'Native GPU .text changed: {description}')
    return dict(
        description=description,
        before=str(before), after=str(after), bytes=len(new_text),
        text_sha256=digest(new_text),
        before_object_sha256=digest(before.read_bytes()),
        after_object_sha256=digest(after.read_bytes()),
        byte_identical=True,
    )


def check_bank_probe(work, output):
    program = work / 'bank-probe'
    subprocess.run([
        'hipcc', '-O3', '-std=c++20', '--offload-arch=gfx90a', '-Wall', '-Wextra',
        str(HERE / 'bank-probe.hip'), '-o', str(program),
    ], check=True)
    subprocess.run(['llvm-objdump-21', '--offloading', str(program)],
                   check=True, stdout=subprocess.DEVNULL)
    object_suffix = '.0.hipv4-amdgcn-amd-amdhsa--gfx90a'
    proof = compare_code_objects(
        Path(str(ROOT / 'temp/native-gemm/bank-probe') + object_suffix),
        Path(str(program) + object_suffix),
        'bank-control kernels',
    )
    with (output / 'bank-check.csv').open('w') as stream:
        subprocess.run(['numactl', '--cpunodebind=0', '--membind=0', str(program)],
                       stdout=stream, check=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=HERE / 'results/final')
    parser.add_argument('--after-source', type=Path, default=HERE / 'results/readable/source')
    parser.add_argument('--after-build-root', type=Path, default=ROOT / 'temp/native-gemm/readable')
    parser.add_argument('--work', type=Path, default=ROOT / 'temp/native-gemm/cleanup/audit')
    parser.add_argument('--output', type=Path, default=HERE / 'results/readable/checks')
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    work = args.work.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh audit output directory')
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    proofs = dict(
        generator=check_generators(baseline, args.after_source.resolve(), work),
        packing=check_packing(baseline, args.after_source.resolve(), work, output),
        native=[],
    )
    for tile in (128, 256):
        proofs['native'].append(compare_code_objects(
            ROOT / f'temp/native-gemm/s32-t{tile}/probe.hsaco',
            args.after_build_root.resolve() / f's32-t{tile}/probe.hsaco',
            f'Selected/control StageK32 tile{tile}',
        ))
    proofs['bank'] = check_bank_probe(work, output)
    (output / 'summary.json').write_text(json.dumps(proofs, indent=2) + '\n')
    print('PASS: eight byte-identical generated headers; packing/reference equivalence; '
          'selected/control/bank GPU .text byte-identical.')
    print(proofs['packing']['result'])


if __name__ == '__main__':
    main()
