#!/usr/bin/env python3
"""Exercise the user's local calculator/venv; retain CLI output and mappings."""
import csv
import io
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[1]
external = root / 'tools/external/amd_matrix_instruction_calculator'
python = external / '.venv/bin/python'
script = external / 'matrix_calculator.py'
out = here / 'results'
out.mkdir(exist_ok=True)

def run(*args):
    return subprocess.check_output([str(python), str(script), *args], text=True)

version = run('--version').strip()
revision = subprocess.check_output(['git', '-C', str(external), 'rev-parse', 'HEAD'], text=True).strip()
for shape, cycles, overlap in [('16x16x16', 32, 28), ('32x32x8', 64, 60)]:
    instruction = 'v_mfma_f32_' + shape + 'f16'
    text = run('-a', 'gfx90a', '-i', instruction, '-d')
    assert f'Execution cycles: {cycles}\n' in text
    assert f'VALU co-execution cycles possible: {overlap}\n' in text
    assert 'GPRs required for A: 2\n' in text
    assert 'GPRs required for B: 2\n' in text
    assert run('-a', 'MI250X', '-i', instruction, '-d') == text
    (out / f'calculator-{16 if cycles == 32 else 32}.txt').write_text(text)

# Independently derive the lane mapping used by the C++ FP64 tile oracle.
# Check all A/B elements and all D outputs, not just a single sample lane.
for matrix in 'ABD':
    text = run('-a', 'gfx90a', '-i', 'v_mfma_f32_32x32x8f16', '-' + matrix, '-M', '--csv')
    (out / f'layout-{matrix}.csv').write_text(text)
    rows = list(csv.reader(io.StringIO(text.split('lane,', 1)[1])))
    assert len(rows) == 65
    seen = set()
    for lane in range(64):
        row = rows[lane + 1]
        assert int(row[0]) == lane
        for reg, cell in enumerate(row[1:]):
            if matrix == 'A':
                coords = lane % 32, 4 * (lane // 32) + reg
            elif matrix == 'B':
                coords = 4 * (lane // 32) + reg, lane % 32
            else:
                coords = 8 * (reg // 4) + 4 * (lane // 32) + reg % 4, lane % 32
            assert cell == f'{matrix}[{coords[0]}][{coords[1]}]', (matrix, lane, reg, cell, coords)
            assert coords not in seen
            seen.add(coords)
    assert len(seen) == (1024 if matrix == 'D' else 256)

text = f'{version}\nupstream revision: {revision}\nPASS gfx90a/MI250X aliases, both FP16 instructions, all 1536 A/B/D element mappings\n'
(out / 'calculator-test.txt').write_text(text)
print(text, end='')
