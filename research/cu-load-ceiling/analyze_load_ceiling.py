#!/usr/bin/env python3
"""Analyze the single-CTA pure-load ceiling probe; verdict vs the 32 B/cycle claim."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/load-ceiling')
MODES={0:'hot',3:'lds',4:'hot2'}
fields=('cycles_per_load','per_cu_median_cycles','body_us_mean','core_MHz','launch_ms')
medians={}; out=[]
for run,rounds,steps in [('run1',9,4096),('run2',9,4096),('small',3,1024),('longer',3,8192)]:
    rows=list(csv.DictReader((folder/f'{run}.csv').open()))
    assert len(rows)==64*rounds,(run,len(rows))
    log=(folder/f'{run}.log').read_text()
    assert 'timed_grid=1x1 block=64*waves' in log and 'no MFMA/conversion' in log
    assert 'PASS eviction reader checksum controls' in log
    assert 'PASS 35 full-width step-64 verify cases' in log
    grouped=defaultdict(list); seen=set()
    for row in rows:
        mode,width,batch,dup,waves,cache,round_=(int(row[k]) for k in ('mode','width','batch','dup','waves','cache','round'))
        ident=(mode,width,batch,dup,waves,cache,round_)
        assert ident not in seen; seen.add(ident)
        assert int(row['steps'])==steps and int(row['eviction_MiB'])==256
        assert int(row['valid'])==1 and int(row['mismatch_count'])==0
        assert MODES[mode]==row['mode_name']
        grouped[mode,width,batch,dup,waves,cache].append(row)
    assert len(grouped)==64 and len({k[:5] for k in grouped})==34,(run,len(grouped))
    for key,samples in sorted(grouped.items()):
        assert len(samples)==rounds
        values={f:median(float(r[f]) for r in samples) for f in fields}
        medians[run,*key]=values
        cyc=values['per_cu_median_cycles']
        out.append([run,steps,*key,*(values[f] for f in fields),
                    key[1]*64/cyc,key[4]*key[1]*64/cyc,key[4]*key[1]*64/cyc/max(key[3],1)])
with (folder/'summary.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['run','steps','mode','width','batch','dup','waves','cache',*fields,
                'wave_B_per_cycle','cu_gross_B_per_cycle','cu_unique_B_per_cycle'])
    w.writerows(out)

def cyc(run,mode,width,batch,dup,waves,cache=0):
    return medians[run,mode,width,batch,dup,waves,cache]['per_cu_median_cycles']
def gross(run,mode,width,batch,dup,waves,cache=0):
    return waves*width*64/cyc(run,mode,width,batch,dup,waves,cache)

L=[]
L.append('One CTA, pure load rings, no MFMA/conversion. Cycles per load per wave (median of per-CU median).')
L.append('B/core-cycle figures are GROSS requested bytes (waves*width*64/cycles) unless labeled unique.')
L.append('')
L.append('1) Hot width x waves sweep (B=8, dup=1), run1/run2 cache-warm | gross B/cyc/CU:')
L.append('W    | waves1 | waves2 | waves4 | waves8')
for width in (4,8,16):
    cells=[]
    for waves in (1,2,4,8):
        c1,c2=(gross(r,0,width,8,1,waves) for r in ('run1','run2'))
        cells.append(f'{c1:6.1f}/{c2:<6.1f}')
    L.append(f'{width:4} | '+' | '.join(cells))
L.append('')
L.append('   cycles/load: W4  '+ ' '.join(f'{cyc("run1",0,4,8,1,w):.1f}' for w in (1,2,4,8))
       + '   W8  '+' '.join(f'{cyc("run1",0,8,8,1,w):.1f}' for w in (1,2,4,8))
       + '   W16 '+' '.join(f'{cyc("run1",0,16,8,1,w):.1f}' for w in (1,2,4,8)))
L.append('')
L.append('2) Ring depth sweep (hot, W4, waves4), cycles/load: '
       + ' '.join(f'B{b}={cyc("run1",0,4,b,1,4):.2f}' for b in (1,2,4,8,16)))
L.append('')
L.append('3) Dup sweep (hot, B=8, waves4): cycles/load vs GROSS/unique B/cyc/CU (run1):')
L.append('W    | ' + ' | '.join(f'dup{d}' for d in (1,2,4,8,16,32,64)))
for width in (4,16):
    c=' | '.join(f'{cyc("run1",0,width,8,d,4):5.1f}' for d in (1,2,4,8,16,32,64))
    g=' | '.join(f'{gross("run1",0,width,8,d,4):5.1f}' for d in (1,2,4,8,16,32,64))
    u=' | '.join(f'{gross("run1",0,width,8,d,4)/d:5.1f}' for d in (1,2,4,8,16,32,64))
    L.append(f'{width:4} cyc | {c}')
    L.append(f'{width:4} gross| {g}')
    L.append(f'{width:4} uniq | {u}')
L.append('')
L.append('4) LDS contrast (B=8, dup=1): ' + ' '.join(
    f'W{w}x{wv}={gross("run1",3,w,8,1,wv):.1f}B/cyc({cyc("run1",3,w,8,1,wv):.1f}cyc)'
    for w,wv in ((4,4),(8,4),(16,4),(16,1))))
L.append('')
L.append('5) Hot2 (data-varying, still L1-hot) vs hot, cycles/load (run1, cache0): '
       + f'W4 {cyc("run1",4,4,8,1,4):.2f} vs {cyc("run1",0,4,8,1,4):.2f}; '
       + f'W16 {cyc("run1",4,16,8,1,4):.2f} vs {cyc("run1",0,16,8,1,4):.2f}')
L.append('')
L.append('6) Cache protocols agree (warm vs evicted+primed): max |delta| cycles/load across all')
L.append('   global cases: ' + f'{max(abs(medians[("run1",)+k+(0,)]["per_cu_median_cycles"]-medians[("run1",)+k+(1,)]["per_cu_median_cycles"]) for k in {(k[1],k[2],k[3],k[4],k[5]) for k in medians if k[0]=="run1" and k[6]==1}):.3f}')
L.append('')
# Verdict against the claim under test.
w16=gross('run1',0,16,8,1,4); w16b=gross('run2',0,16,8,1,4)
w4=gross('run1',0,4,8,1,8); w8=gross('run1',0,8,8,1,4); lds=gross('run1',3,16,8,1,4)
dupflat=max(abs(cyc('run1',0,16,8,d,4)-cyc('run1',0,16,8,1,4)) for d in (2,4,8,16,32,64))
L.append('VERDICT vs "each CU can receive at most 32 bytes per cycle":')
L.append(f'- Hot dwordx4 (W16) at 4 waves sustains {w16:.1f}/{w16b:.1f} B/core-cycle gross (~2x the claim);')
L.append(f'  hot dword at 8 waves sustains {w4:.1f} B/core-cycle; LDS b128 delivers {lds:.1f} B/core-cycle.')
L.append(f'- Hot dwordx2 (W8) saturates near {w8:.1f} B/core-cycle: dwordx2 and dwordx4 take the SAME')
L.append(f'  ~{cyc("run1",0,16,8,1,4):.0f} cycles per instruction at 4 waves, so the limiter is per-instruction, not per-byte.')
L.append(f'- Dup sweep: W16 cycles/load change by only {dupflat:.1f} cycles between 1024 B and 16 unique B')
L.append('  per instruction -> the hot ceiling is NOT unique-byte delivery.')
L.append('- The fp8-w8a16 fixed-probe "32 B/core-cycle" is therefore not a universal CU receive ceiling;')
L.append('  it is consistent with a per-instruction global-load return limit (~1 wide load per ~14-16 CU')
L.append('  cycles, ~1 dword load per ~4-5 cycles) acting on its 8-loads-per-step pattern, plus its')
L.append('  conversion/copy VALUs. No hardware counters were used; the limiting unit is not identified.')
text='\n'.join(L)+'\n'
(folder/'summary.txt').write_text(text)
print(text)
