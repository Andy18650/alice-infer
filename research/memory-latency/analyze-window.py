#!/usr/bin/env python3
"""Keep unprofiled throughput separate from same-pass VMEM lifetime/occupancy."""
import argparse,collections,csv,hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
def save(name,rows):
 with (out/name).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
raw=[];manifest=[]
for path in map(Path,a.runs):
 assert (path/'complete').exists()
 for r in csv.DictReader((path/'results.csv').open()):raw.append({'run':path.name,**r})
 manifest.append({'run':str(path),'binary_sha256':hashlib.sha256((path/'measured-window').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((path/'source/window.hip').read_bytes()).hexdigest(),'burst_header_sha256':hashlib.sha256((path/'source/window-bursts.hpp').read_bytes()).hexdigest(),'command':(path/'command.txt').read_text().strip(),'device':(path/'run.log').read_text().strip()})
assert len({r['binary_sha256']for r in manifest})==1
save('window-samples.csv',raw)
groups=collections.defaultdict(list)
for r in raw:
 assert r['private_bytes']=='0'and r['lds_bytes']=='0'
 groups[tuple(int(r[k])for k in ['width_dwords','depth','grid'])].append(r)
medians=[]
for (w,d,g),rs in sorted(groups.items()):
 first=rs[0];vals=[float(r['requested_GBps'])for r in rs]
 r={'width_dwords':w,'depth':d,'grid':g,'samples':len(rs),'nominal_window_bytes':int(first['nominal_window_bytes']),'resident_window_upper_bytes':int(first['resident_window_upper_bytes']),'registers':int(first['registers']),'median_GBps':statistics.median(vals),'min_GBps':min(vals),'max_GBps':max(vals)}
 for run in sorted({x['run']for x in rs}):r['GBps_'+run]=statistics.median(float(x['requested_GBps'])for x in rs if x['run']==run)
 medians.append(r)
save('window-medians.csv',medians)
profile_rows=[]
for path in map(Path,a.profiles):
 assert (path/'complete').exists()
 captures=list(path.rglob('*counter_collection.csv'));traces=list(path.rglob('*kernel_trace.csv'))
 assert len(captures)==len(traces)==1,'ambiguous/stale profiler outputs'
 capture,trace=captures[0],traces[0];captured={}
 assert hashlib.sha256((path/'measured-window').read_bytes()).hexdigest()==manifest[0]['binary_sha256']
 for r in csv.DictReader(capture.open()):
  rec=captured.setdefault(r['Dispatch_Id'],{'metadata':r,'counters':{}})
  assert r['Counter_Name']not in rec['counters'];rec['counters'][r['Counter_Name']]=float(r['Counter_Value'])
 assert len(captured)==80
 ordinal=collections.Counter()
 for r in csv.DictReader(trace.open()):
  if 'window_entry<'not in r['Kernel_Name']:continue
  ordinal[r['Kernel_Name']]+=1
  if r['Dispatch_Id']in captured:assert ordinal[r['Kernel_Name']]%4==0,'not fourth launch of case'
 seen=set()
 for rec in captured.values():
  r,c=rec['metadata'],rec['counters'];w,d=map(int,re.search(r'window_entry<(\d+), (\d+)>',r['Kernel_Name']).groups());grid=int(r['Grid_Size'])//256
  key=(w,d,grid);assert key not in seen;seen.add(key)
  rs=groups[key];size=int(rs[0]['bytes']);waves=grid*4;payload=64*w*4
  load_instructions=size//payload
  assert c['SQ_WAVES']==waves
  assert c['SQ_INSTS_VMEM']==load_instructions+waves,'loads + one checksum store per wave'
  assert c['TCC_EA_RDREQ_32B']==0
  external_bytes=64*c['TCC_EA_RDREQ'];assert 0.999<external_bytes/size<1.001
  assert c['TCC_EA_RDREQ_DRAM']/c['TCC_EA_RDREQ']>0.999
  dt=int(r['End_Timestamp'])-int(r['Start_Timestamp']);cycles=c['GRBM_GUI_ACTIVE'];lat=c['VmemLatency']
  slots=lat*c['SQ_INSTS_VMEM']/cycles
  assert 0<slots<=waves*d*1.01,'occupancy outside nominal bound'
  profile_rows.append({'run':path.name,'width_dwords':w,'depth':d,'grid':grid,'waves':waves,'payload_per_load':payload,'nominal_window_bytes':waves*d*payload,'load_instructions':load_instructions,'all_vmem_instructions':c['SQ_INSTS_VMEM'],'profile_kernel_ns':dt,'profile_requested_GBps':size/dt,'external_read_bytes':external_bytes,'external_read_GBps':external_bytes/dt,'grbm_cycles':cycles,'effective_MHz':cycles/dt*1000,'vmem_lifetime_cycles':lat,'vmem_lifetime_ns':lat*dt/cycles,'mean_outstanding_vmem_instructions':slots,'vmem_byte_equivalent_window':slots*payload,'store_instruction_fraction':waves/c['SQ_INSTS_VMEM']})
 manifest.append({'profile':str(path),'binary_sha256':hashlib.sha256((path/'measured-window').read_bytes()).hexdigest(),'counter_config_sha256':hashlib.sha256((path/'input.yaml').read_bytes()).hexdigest(),'command':(path/'command.txt').read_text().strip()})
 assert seen==set(groups)
save('window-profile-samples.csv',profile_rows)
pgroups=collections.defaultdict(list)
for r in profile_rows:pgroups[r['width_dwords'],r['depth'],r['grid']].append(r)
profile_medians=[]
for (w,d,g),rs in sorted(pgroups.items()):
 r={'width_dwords':w,'depth':d,'grid':g,'samples':len(rs)}
 for field in ['profile_requested_GBps','external_read_GBps','vmem_lifetime_cycles','vmem_lifetime_ns','mean_outstanding_vmem_instructions','vmem_byte_equivalent_window','effective_MHz']:
  vs=[x[field]for x in rs];r[field+'_median']=statistics.median(vs);r[field+'_min']=min(vs);r[field+'_max']=max(vs)
 profile_medians.append(r)
save('window-profile-medians.csv',profile_medians)
(out/'window-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in medians:
 if r['width_dwords']==4 and r['grid']==104:print('unprofiled',r['depth'],r['nominal_window_bytes']/1024,'KiB',r['median_GBps'],'GB/s')
for r in profile_medians:
 if r['width_dwords']==4 and r['grid']==104:print('profile',r['depth'],'latency ns',r['vmem_lifetime_ns_median'],'equivalent KiB',r['vmem_byte_equivalent_window_median']/1024)
print('Validated',len(raw),'unprofiled samples and',len(profile_rows),'same-pass counter samples')
