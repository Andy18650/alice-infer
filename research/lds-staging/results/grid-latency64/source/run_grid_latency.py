#!/usr/bin/env python3
"""Grid latency sweep: preserve every interval in compressed CSV, no filtering."""
import argparse,array,csv,gzip,json,re,shlex,shutil,statistics,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/grid-latency');p.add_argument('--output',type=Path,default=here/'results/grid-latency64');a=p.parse_args()
build=a.build.resolve();out=a.output.resolve();counts=(1,8,16,32,64,128,256);batches=64
if out.exists() and any(out.iterdir()):p.error('choose a fresh output directory')
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','build','raw'):(out/sub).mkdir(parents=True,exist_ok=True)
for name in ('grid-latency.hip','generate-latency.py','build-grid-latency.sh','inspect_latency.py','run_grid_latency.py'):shutil.copy2(here/name,out/'source'/name)
for name in ('latency-records.hpp','compiler-flags.txt','hashes.txt','metadata.txt','timed-samples.txt','inventory.json'):shutil.copy2(build/name,out/'build'/name)
(out/'invocation.json').write_text(json.dumps(dict(build=str(build),ctas=counts,batches=batches,source_bytes_per_cta=1048576,placement='numactl CPU+memory node0',filtering=False),indent=2)+'\n')
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
def stats(v):
    q=statistics.quantiles(v,n=100,method='inclusive')
    return dict(n=len(v),min=min(v),max=max(v),median=statistics.median(v),mean=statistics.mean(v),stdev=statistics.pstdev(v),p05=q[4],p25=q[24],p75=q[74],p95=q[94],p99=q[98],iqr=q[74]-q[24],range=max(v)-min(v))
rows=[];by_batch=[]
with (out/'commands.txt').open('w') as commands:
    for repeat in (1,2):
        for ctas in counts if repeat==1 else reversed(counts):
            for mode in ('shared','private') if repeat==1 else ('private','shared'):
                name=f'{mode}-n{ctas}-r{repeat}'
                cmd=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),'--mode',mode,'--ctas',str(ctas),'--batches',str(batches),'--warmup-ms','200']
                commands.write(shlex.join(cmd)+'\n');commands.flush();values=array.array('I')
                with (out/'raw'/(name+'.log')).open('w') as err,subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err,text=True) as proc,gzip.open(out/'raw'/(name+'.csv.gz'),'wt',compresslevel=6) as dst:
                    reader=csv.DictReader(proc.stdout);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames,lineterminator='\n');writer.writeheader()
                    for index,row in enumerate(reader):
                        b,c,w,s=(int(row[k]) for k in ('batch','cta','wave','sample'))
                        assert ((b*ctas+c)*4+w)*64+s==index and int(row['ctas'])==ctas and row['mode']==mode
                        assert int(row['source_offset'])==(b%16)*16384
                        values.append(int(row['cycles']));writer.writerow(row)
                    assert proc.wait()==0,name
                assert len(values)==ctas*batches*256,(name,len(values))
                log=(out/'raw'/(name+'.log')).read_text();assert 'final per-CTA LDS checks and all record/check guards pass' in log
                l2=int(re.search(r'L2_bytes=(\d+)',log)[1]);cus=int(re.search(r'CUs=(\d+)',log)[1])
                row=dict(mode=mode,ctas=ctas,repeat=repeat,CUs=cus,L2_bytes=l2,source_bytes=ctas*1048576,touched_per_launch=ctas*65536,**stats(values));rows.append(row)
                span=ctas*256
                for b in range(batches):by_batch.append(dict(mode=mode,ctas=ctas,repeat=repeat,batch=b,**stats(values[b*span:(b+1)*span])))
                for filename,data in (('summary.csv',rows),('per-batch.csv',by_batch)):
                    with (out/filename).open('w') as f:writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
                print(f"{name}: n={len(values)} min/median/mean/p95/max={row['min']}/{row['median']}/{row['mean']:.2f}/{row['p95']}/{row['max']}",flush=True)
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(f"COMPLETE {sum(r['n'] for r in rows)} intervals, all final per-CTA LDS checks/guards pass; no filtering.")
