#!/usr/bin/env python3
"""Summarize unloaded dependent VMEM measurements, keeping timing types separate."""
import argparse,collections,csv,hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
p.add_argument('--counter-log');p.add_argument('--counter-csv');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
rows=[];manifest=[]
for folder in map(Path,a.runs):
 assert (folder/'complete').exists(),folder
 for r in csv.DictReader((folder/'results.csv').open()):
  assert float(r['cycles_per_step'])>0 and float(r['ns_per_step'])>0
  rows.append({'run':folder.name,**r})
 manifest.append({'run':str(folder),'binary_sha256':hashlib.sha256((folder/'measured-probe').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((folder/'source/probe.hip').read_bytes()).hexdigest(),'command':(folder/'command.txt').read_text().strip(),'device':(folder/'run.log').read_text().strip()})
assert len({m['binary_sha256']for m in manifest})==1,'mixed executables'
assert len({m['source_sha256']for m in manifest})==1,'mixed sources'
def write_csv(path,rs):
 with path.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
write_csv(out/'samples.csv',rows)
groups=collections.defaultdict(list)
for r in rows:groups[(r['mode'],r['policy'],int(r['working_bytes']))].append(r)
summaries=[]
for (mode,policy,size),rs in sorted(groups.items()):
 d={'mode':mode,'policy':policy,'working_bytes':size,'samples':len(rs)}
 for field in ['cycles_per_step','ns_per_step','effective_MHz','single_cycles','empty_cycles']:
  vs=[float(r[field])for r in rs]
  for op,func in [('median',statistics.median),('min',min),('max',max)]:d[field+'_'+op]=func(vs)
 if mode!='dependent-add':
  d['single_minus_empty_cycles_median']=statistics.median(float(r['single_cycles'])-float(r['empty_cycles'])for r in rs)
 else:d['single_minus_empty_cycles_median']=''
 for run in sorted({r['run']for r in rs}):d['step_cycles_'+run]=statistics.median(float(r['cycles_per_step'])for r in rs if r['run']==run)
 summaries.append(d)
write_csv(out/'medians.csv',summaries)
(out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
if a.counter_log and a.counter_csv:
 lines=Path(a.counter_log).read_text().splitlines()
 header='mode,policy,working_bytes,node_bytes,nodes,round,iterations,cycles_per_step,ns_per_step,effective_MHz,single_cycles,empty_cycles,hwid,final_index'
 data=[line for line in lines if line.startswith(('lane-dword,','wave-dword,','wave-dwordx4,','dependent-add,'))]
 bykey=collections.defaultdict(list)
 for r in csv.DictReader([header]+data):bykey[r['mode'],r['policy']].append(r)
 captured={}
 for r in csv.DictReader(Path(a.counter_csv).open()):
  g=captured.setdefault(int(r['Dispatch_Id']),{'name':r['Kernel_Name'],'counters':{}})
  assert r['Counter_Name']not in g['counters'];g['counters'][r['Counter_Name']]=float(r['Counter_Value'])
 index=collections.Counter();joined=[]
 for dispatch,g in sorted(captured.items()):
  width,single,glc,control=re.search(r'chase<(\d+), (true|false), (true|false), (true|false)>',g['name']).groups()
  mode='dependent-add'if control=='true'else'lane-dword'if single=='true'else'wave-dwordx4'if width=='4'else'wave-dword'
  policy='control'if control=='true'else'glc'if glc=='true'else'default';key=(mode,policy)
  info=bykey[key][index[key]];index[key]+=1;c=g['counters']
  assert c['SQ_WAVES']==1
  expected=4 if control=='true'else int(info['nodes'])+int(info['iterations'])+1+4
  assert c['SQ_INSTS_VMEM']==expected,(key,c['SQ_INSTS_VMEM'],expected)
  joined.append({'mode':mode,'policy':policy,'working_bytes':int(info['working_bytes']),'nodes':int(info['nodes']),'iterations':int(info['iterations']),**c})
 assert len(joined)==len(data)==63
 write_csv(out/'cache-counters.csv',joined)
 print('Validated 63 counter records: one wave and exact warm/single/chain/store VMEM counts')
print('Samples:',len(rows),'groups:',len(summaries))
for r in summaries:
 if r['mode']=='wave-dword'and r['policy']=='default':
  print(r['working_bytes'],'step cycles/ns',r['cycles_per_step_median'],r['ns_per_step_median'],'single/empty cycles',r['single_cycles_median'],r['empty_cycles_median'])
