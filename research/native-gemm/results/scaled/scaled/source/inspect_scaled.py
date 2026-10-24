#!/usr/bin/env python3
"""Actual ISA/register audit for compact/all-architected scaled variants."""
import argparse
import json
import re
from pathlib import Path
from inspect_code import parse_instructions, find_hot_loop, opcode_counts, integer_field, require, MFMA


def audit(output):
    header = (output/'compute.hpp').read_text()
    scaled = 'kScaled=true' in header
    instructions = parse_instructions((output/'probe.isa').read_text())
    start = next(i for i,(_,op) in enumerate(instructions) if op.startswith('s_memtime'))
    end = next(i for i in range(start+1,len(instructions)) if instructions[i][1].startswith('s_memtime'))
    body = [op for _,op in instructions[start:end+1]]
    loop = find_hot_loop(instructions,128)
    counts = opcode_counts(loop)
    require(counts[MFMA] == 128, 'Wrong MFMA count')
    require(counts['v_fma_f32'] == (128 if scaled else 0), 'Wrong scale FMA count')
    require(counts['s_barrier'] == 16 and counts['ds_read_b128'] == 64,
            'Changed barrier/raw-read budget')
    require(counts['global_load_dwordx4'] == 24 and counts['ds_write_b128'] == 32,
            'Changed global/LDS payload budget')
    require(not any(re.search(r'accvgpr|scratch_|writelane|readlane',op) for op in body),
            'Accumulator transfer/spill in timed body')
    require(all(' v[' in op and ', v[' in op for op in loop if op.startswith(MFMA)),
            'MFMA not on architected VGPRs')
    notes = (output/'metadata.txt').read_text()
    fields = {key:integer_field(notes,r'\.'+key+r':\s*(\d+)')
              for key in ('vgpr_count','sgpr_count','agpr_count','private_segment_fixed_size')}
    require(fields['private_segment_fixed_size']==0 and fields['agpr_count']==0,
            'Spill or unexpected AGPR allocation')
    require(fields['vgpr_count'] <= 256, 'Exceeded architected budget')
    fields.update(tile=128,stage_k=32,kind='gemm',lds_bytes=16384,scaled=scaled,
                  instructions_per_loop=dict(counts))
    (output/'inventory.json').write_text(json.dumps(fields,indent=2)+'\n')
    (output/'loop.txt').write_text('\n'.join(loop)+'\n')
    (output/'body.txt').write_text('\n'.join(body)+'\n')
    print(fields)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    audit(parser.parse_args().output)
