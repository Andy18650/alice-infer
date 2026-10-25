#!/usr/bin/env python3
"""Read-only artifact/clock verification, independent of the C++ clock summary."""
import argparse
import csv
import hashlib
import json
import re
import statistics
import struct
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clock_check(blob,row,rate):
    blocks=int(row['ctas'])
    assert len(blob)==blocks*128,'wrong clock capture length'
    values=list(struct.iter_unpack('<4Q',blob))
    starts=[];ends=[];ratios=[]
    for block in range(blocks):
        waves=values[block*4:block*4+4]
        assert all(ce>cb and se>sb for cb,ce,sb,se in waves),'nonmonotonic clock'
        cb=min(w[0] for w in waves);ce=max(w[1] for w in waves)
        sb=min(w[2] for w in waves);se=max(w[3] for w in waves)
        ratios.append((ce-cb)*rate/((se-sb)*1000))
        starts.append(sb);ends.append(se)
    body=(max(ends)-min(starts))*1000/rate
    assert body<=float(row['gemm_us'])*1.05+10,'span outside independent HIP event'
    assert abs(body-float(row['grid_body_us']))<0.000002,'raw/body CSV mismatch'
    assert abs(statistics.median(ratios)-float(row['median_core_MHz']))<0.001,'raw/ratio CSV mismatch'


def verify_results(output,check_manifest=True):
    if check_manifest:
        manifest=json.loads((output/'manifest.json').read_text())
        for name,sha in manifest.items():assert digest(output/name)==sha,('artifact changed',name)
    provenance=json.loads((output/'provenance.json').read_text())
    source_checks=0
    for layout,build in provenance['builds'].items():
        for line in (output/'builds'/layout/'hashes.txt').read_text().splitlines():
            sha,name=line.split('  ',1)
            if '/research/' in name:
                path=output/'builds'/layout/'source'/'research'/name.split('/research/',1)[1]
            elif Path(name).name in ('compute.hpp',):path=output/'builds'/layout/Path(name).name
            else:path=Path(name) # Measured binaries/code objects remain in durable gitignored temp/.
            assert digest(path)==sha,('recorded source/native hash mismatch',path)
            source_checks+=1
    captures=negative=rows=0
    for file in sorted(output.glob('*/*/*/*/stdout.csv')):
        samples=list(csv.DictReader(file.open()))
        log=file.with_name('stderr.txt').read_text()
        rate=int(re.search(r'wall_clock_khz=(\d+)',log)[1])
        assert 'COMPLETE' in log and 'GPU A pack FULL' in log
        for row in samples:
            assert row['valid']=='1'
            prefix=f"clocks-{row['protocol']}-r{row['round']}.bin"
            blob=file.with_name(prefix).read_bytes();clock_check(blob,row,rate)
            # Reproduce the old default-V_PK_MOV bug. A correlated low-word
            # duplication can leave clock RATIOS plausible, but not event span.
            words=struct.unpack('<'+'Q'*(len(blob)//8),blob)
            corrupt=struct.pack('<'+'Q'*len(words),*((x&0xffffffff)*0x100000001 for x in words))
            try:clock_check(corrupt,row,rate)
            except AssertionError:negative+=1
            else:raise AssertionError('duplicated-low-word negative accepted')
            captures+=1
        rows+=len(samples)
    assert rows==provenance['checked_launches']
    return dict(checked_launches=rows,hashed_processes=provenance['process_count'],
                raw_clock_captures=captures,duplicated_low_word_rejections=negative,
                source_native_hash_checks=source_checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(verify_results(a.output),indent=2))
