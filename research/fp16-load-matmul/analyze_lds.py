#!/usr/bin/env python3
"""LDS stage tables and address-bank model; no matrix-output validation."""
from collections import Counter
import csv,json
from pathlib import Path

root=Path(__file__).resolve().parent/'results/lds'
rows=list(csv.DictReader((root/'summary.csv').open()))
index={(int(r['depth']),int(r['lds_stride']),r['mode'],int(r['loads']),int(r['wait'])):r for r in rows}
lines=['LDS, U32, K32768, one CTA/four waves. CTA makespan cycles/step.',
       'Waited (lgkmcnt), packed stride1; pooled medians over 18 samples/case.']
for mode in ('lds-hot','lds-sweep'):
    lines+=['',mode,'D | 1*b128 | 2*b64 | 4*b32']
    for d in (1,2,3,4,8):
        lines.append(str(d)+' | '+' | '.join(f"{float(index[d,1,mode,n,1]['cta_cycles_per_step']):.3f}" for n in (1,2,4)))
lines+=['','Run spread to retain (scalar4):',
        'D1 hot: 96.893 / 100.892; D4 sweep: 69.424 / 65.928 cycles/step.',
        'Do not interpret pooled medians as a precise monotonic latency curve.',
        '','Deliberate stride4 bank concentration, hot waited:']
for d in (1,4):
    lines.append(str(d)+' | '+' | '.join(f"{float(index[d,4,'lds-hot',n,1]['cta_cycles_per_step']):.3f}" for n in (1,2,4)))
# Aggregate address pressure, not a claim about undocumented per-instruction
# DS service phases. Count distinct dwords to distinguish conflicts from broadcast.
banks=[]
for stride in (1,2,4,8):
    for loads in (1,2,4):
        addresses=[]
        lane_words=4//loads
        for wave in range(4):
            for lane in range(64):
                for plane in range(loads):
                    for word in range(lane_words):
                        addresses.append(wave*256*stride+lane*lane_words*stride+plane*(256//loads)*stride+word)
        counts=Counter(word%32 for word in set(addresses))
        banks.append(dict(stride=stride,loads=loads,distinct_dwords=len(set(addresses)),
                          active_banks=len(counts),max_distinct_dwords_per_bank=max(counts.values()),
                          per_bank={str(i):counts[i] for i in range(32)}))
(root/'bank-model.json').write_text(json.dumps(banks,indent=2)+'\n')
lines+=['','Bank model (all four waves, one pair):',
        'stride1: 1024 distinct dwords, 32 active banks, 32 dwords/bank.',
        'stride4: same 1024 distinct dwords, 8 active banks, 128 dwords/bank.',
        'This models aggregate bank pressure, not a direct conflict counter or proof of wide-read phase mapping.',
        'Depth = raw A+B fragment slots, PLUS two ready operand banks.',
        'D2 reaches the hot ~64-cycle rate for all shapes; sweep wide reads also near ~65.',
        'Sweep scalar4 retains a non-monotonic ~66..69-cycle waited floor; no-wait is ~65.',
        'LDS queue priming is inside timing; timer lgkmcnt(0) waits cannot drain it before the loop.',
        'LGKM immediate 15 is the all-ones/no-wait setting. Stronger wait cap14 is reported for deep rings.',
        'D8/scalar4 requested28 is capped14; D1..4 stage decisions are uncapped.',
        'No VRAM-to-LDS transfer in the timed path; LDS is filled directly before timing. No numerical tests.']
(root/'stage-summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
