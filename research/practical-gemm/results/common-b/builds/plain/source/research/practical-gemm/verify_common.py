#!/usr/bin/env python3
"""Read-only shared-B evidence/hash and independent raw-clock verification."""
import argparse
import csv
import json
import re
import statistics
import struct
from pathlib import Path
from verify import digest


def clock_check(blob,row,rate):
    ctas=int(row['ctas']);waves=int(row['waves_per_cta'])
    assert waves in (1,4) and len(blob)==ctas*waves*32
    clocks=list(struct.iter_unpack('<4Q',blob));starts=[];ends=[];ratios=[]
    for i in range(ctas):
        group=clocks[i*waves:(i+1)*waves]
        assert all(ce>cb and se>sb for cb,ce,sb,se in group)
        cb=min(x[0] for x in group);ce=max(x[1] for x in group)
        sb=min(x[2] for x in group);se=max(x[3] for x in group)
        starts.append(sb);ends.append(se);ratios.append((ce-cb)*rate/((se-sb)*1000))
    body=(max(ends)-min(starts))*1000/rate
    assert body<=float(row['event_us'])*1.05+10,'body outside independent full event'
    assert abs(body-float(row['grid_body_us']))<0.000002
    assert abs(statistics.median(ratios)-float(row['median_core_MHz']))<0.001


def verify_results(output,check_manifest=True):
    if check_manifest:
        for name,sha in json.loads((output/'manifest.json').read_text()).items():
            assert digest(output/name)==sha,('artifact changed',name)
    provenance=json.loads((output/'provenance.json').read_text());hashes=0
    for fmt in provenance['builds']:
        for line in (output/'builds'/fmt/'hashes.txt').read_text().splitlines():
            sha,name=line.split('  ',1)
            if '/research/' in name:path=output/'builds'/fmt/'source'/'research'/name.split('/research/',1)[1]
            elif Path(name).name in ('compute.hpp','kernel.hpp'):path=output/'builds'/fmt/Path(name).name
            else:path=Path(name)
            assert digest(path)==sha,('source/native hash mismatch',path);hashes+=1
    captures=negative=processes=0
    for file in sorted(output.glob('*/*/*/*/stdout.csv')):
        log=file.with_name('stderr.txt').read_text()
        assert 'PRIMITIVES 65024 exact' in log and 'INTEROP same B pointer GEMM+decode PASS' in log and 'COMPLETE' in log
        rate=int(re.search(r'wall_clock_khz=(\d+)',log)[1]);processes+=1
        for row in csv.DictReader(file.open()):
            assert row['valid']=='1'
            blob=file.with_name(f"clocks-r{row['round']}.bin").read_bytes();clock_check(blob,row,rate)
            data=struct.unpack('<'+'Q'*(len(blob)//8),blob)
            corrupt=struct.pack('<'+'Q'*len(data),*((x&0xffffffff)*0x100000001 for x in data))
            try:clock_check(corrupt,row,rate)
            except AssertionError:negative+=1
            else:raise AssertionError('duplicated-low-word negative accepted')
            captures+=1
    assert processes==provenance['process_count'] and captures==provenance['checked_timed_invocations']
    return dict(checked_timed_invocations=captures,checked_interop_invocations=processes*2,
                hashed_processes=processes,full_layout_roundtrip_processes=processes,
                finite_conversion_transpose_primitive_processes=processes,
                raw_clock_captures=captures,duplicated_low_word_rejections=negative,source_native_hash_checks=hashes)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(verify_results(a.output),indent=2))
