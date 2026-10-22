#!/usr/bin/env python3
"""Focused native-GEMM correctness +matched full-problem tile comparison."""
import argparse,csv,json,shlex,shutil,statistics,struct,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=here/'results/final');p.add_argument('--build128',type=Path,default=root/'temp/native-gemm/s32-t128');p.add_argument('--build256',type=Path,default=root/'temp/native-gemm/s32-t256');a=p.parse_args()
out=a.output.resolve();builds={128:a.build128.resolve(),256:a.build256.resolve()}
if out.exists() and any(out.iterdir()):p.error('choose a fresh results directory')
for build in builds.values():subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','builds','verification','timing','clocks'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('probe.hip','data.hpp','generate.py','build.sh','inspect_code.py','run.py'):shutil.copy2(here/name,out/'source'/name)
for tile,build in builds.items():
 d=out/'builds'/str(tile);d.mkdir()
 for name in ('compute.hpp','compiler-flags.txt','hashes.txt','inventory.json','loop.txt','metadata.txt'):shutil.copy2(build/name,d/name)
problems=((1664,1024,8192),(3328,2048,8192),(4096,6656,8192),(4096,6656,32768))
(out/'invocation.json').write_text(json.dumps(dict(builds={str(k):str(v) for k,v in builds.items()},problems=problems,stage_k=32,swizzle=0,placement='numactl CPU+memory node0',libraries='none',packing='CPU outside GEMM timing'),indent=2)+'\n')
rows=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
 def run(tile,m,n,k,repeat=0,sa=1729,sb=2718):
  timing=repeat>0;folder='timing' if timing else 'verification';stem=f't{tile}-m{m}-n{n}-k{k}-r{repeat}-a{sa}-b{sb}'
  cmd=['numactl','--cpunodebind=0','--membind=0',str(builds[tile]/'probe'),'--m',str(m),'--n',str(n),'--k',str(k),'--rounds','9' if timing else '1','--warmup-ms','200' if timing else '0','--swizzle','0','--seed-a',str(sa),'--seed-b',str(sb)]
  if timing:cmd+=['--stamps-prefix',str(out/'clocks'/stem)]
  commands.write(shlex.join(cmd)+'\n');commands.flush()
  with (out/folder/(stem+'.csv')).open('w') as f,(out/folder/(stem+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
  with (out/folder/(stem+'.csv')).open() as f:data=list(csv.DictReader(f))
  assert len(data)==(9 if timing else 1)
  assert 'COMPLETE all reference outputs/finite C/guards exact' in (out/folder/(stem+'.log')).read_text()
  for x in data:
   assert x['kind']=='gemm' and int(x['tile'])==tile and x['stage_k']=='32' and x['swizzle']=='0'
   assert x['valid']=='1' and x['mismatches']==x['nonfinite']==x['guards']=='0'
   assert int(x['lds_bytes'])==tile*128 and int(x['ctas'])==((m+tile-1)//tile)*((n+tile-1)//tile)
   if timing:
    b=(out/'clocks'/(stem+f"-r{x['round']}.bin")).read_bytes();t=struct.unpack('<'+'Q'*(len(b)//8),b);blocks=int(x['ctas']);assert len(t)==blocks*16
    wb=min(t[i*16+w*4+2] for i in range(blocks) for w in range(4));we=max(t[i*16+w*4+3] for i in range(blocks) for w in range(4));tick_us=float(x['grid_body_us'])/(we-wb)
    mhz=[];shares=[]
    for i in range(blocks):
     v=t[i*16:(i+1)*16];cycles=max(v[w*4+1] for w in range(4))-min(v[w*4] for w in range(4));wall=max(v[w*4+3] for w in range(4))-min(v[w*4+2] for w in range(4))
     mhz.append(cycles/(wall*tick_us));shares.append((2*tile*tile*int(x['kp']))/(cycles*1024))
    assert abs(statistics.median(mhz)-float(x['median_core_MHz']))<0.002
    assert abs(statistics.median(shares)-float(x['median_cta_share_of_cu_peak']))<1e-8
    eff=float(x['useful_TFLOPs'])/(104*1024*statistics.median(mhz)/1e6)
    assert abs(eff-float(x['global_clock_adjusted_efficiency']))<1e-6
  row=dict(kind=folder,tile=tile,m=m,n=n,k=k,repeat=repeat,samples=len(data),full_reference=data[0]['full_reference'],reference_outputs=data[0]['reference_outputs'],ctas=data[0]['ctas'],resident_ctas_per_cu=data[0]['resident_ctas_per_cu'])
  for key in ('useful_TFLOPs','executed_TFLOPs','event_us','grid_body_us','median_core_MHz','median_cta_share_of_cu_peak','global_clock_adjusted_efficiency','pack_ms'):row[key]=statistics.median(float(x[key]) for x in data)
  rows.append(row)
  with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  if timing:print(f"{stem}: {row['useful_TFLOPs']:.3f} TFLOP/s, clock={row['median_core_MHz']:.1f}MHz, global efficiency={row['global_clock_adjusted_efficiency']:.3f}",flush=True)
 cases=((1,1,0),(1,1,1),(17,23,7),(31,35,8),(64,64,31),(127,129,32),(128,128,33),(129,131,63),(129,131,64),(129,131,65),(255,257,127),(257,259,129),(128,128,512),(257,129,1024),(17,19,65535),(17,19,65536))
 for tile in builds:
  for m,n,k in cases:run(tile,m,n,k)
  run(tile,129,131,65,sa=1730);run(tile,129,131,65,sb=2719)
 jobs=[(tile,m,n,k) for m,n,k in problems for tile in builds]
 for repeat in (1,2):
  for tile,m,n,k in jobs if repeat==1 else reversed(jobs):run(tile,m,n,k,repeat)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(f"COMPLETE {sum(r['samples'] for r in rows)} verified launches; small full references and large sampled references/finite all-C/guards pass.")
