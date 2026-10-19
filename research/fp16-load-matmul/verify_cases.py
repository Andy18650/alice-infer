#!/usr/bin/env python3
"""Full-output verification, including direct launches of original code objects."""
import argparse,csv,hashlib,itertools,json,re,shlex,shutil,subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/verification')
p.add_argument('--resume',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
casesdir=a.output/'cases';casesdir.mkdir(exist_ok=True)
root=Path(__file__).resolve().parents[2]
base=root/'temp/fp16-load-matmul'
modes=('hot','l2','stream','lds-hot','lds-sweep')
configs=[(1,32,1),(2,32,1),(3,32,1),(4,32,1),(8,32,1),(1,32,4),(4,32,4),(16,1,1),(3,8,1)]
patterns=[('legacy',1729,2718),('hash',1729,2718),('hash',1730,2718),('hash',1729,2719),('hash',1730,2719)]
jobs=[]
for d,u,s in configs:
    selected=modes if (d,u,s) in ((8,32,1),(16,1,1),(3,8,1)) else modes[3:]
    for mode,loads in itertools.product(selected,(1,2,4)):
        for pattern,sa,sb in patterns:jobs.append(('new',d,u,s,mode,loads,1,pattern,sa,sb,4096,0))
        # Original measured kernel, original input pattern. For globals also
        # exercise record-varying data: the old ABI takes an external buffer.
        if mode in modes[:3] or (d,u) not in ((16,1),(3,8)):
            jobs.append(('original',d,u,s,mode,loads,1,'legacy',1729,2718,4096,0))
        if mode in modes[:3]:jobs.append(('original',d,u,s,mode,loads,1,'hash',1729,2718,4096,0))
# End midway through the register-rotation cycle; odd cyclic window size.
for d,u,steps in ((16,1,17),(3,8,40)):
    for mode,loads in itertools.product(modes,(1,2,4)):
        jobs.append(('edge',d,u,1,mode,loads,1,'hash',1730,2719,steps,7 if mode in ('l2','lds-sweep') else 0))
metadata={};copied=set();summary=[]
with (a.output/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (a.output/'commands.txt').open('w') as commands:
    for idx,job in enumerate(jobs):
        backend,d,u,s,mode,loads,wait,pattern,sa,sb,steps,records=job
        folder=base/f'd{d}-u{u}-lds{s}-verified'
        tag=f'{backend}-d{d}-u{u}-s{s}-{mode}-l{loads}-w{wait}-{pattern}-a{sa}-b{sb}-n{steps}'
        out=casesdir/(tag+'.csv');log=casesdir/(tag+'.log')
        if folder not in copied:
            dest=a.output/'builds'/folder.name;dest.mkdir(parents=True,exist_ok=True)
            for file in ('hashes.txt','compiler-flags.txt','inventory.json'):shutil.copy2(folder/file,dest/file)
            copied.add(folder)
        cmd=['numactl','--cpunodebind=0','--membind=0',str(folder/'probe'),
             '--mode',mode,'--loads',str(loads),'--wait',str(wait),'--pattern',pattern,
             '--seed-a',str(sa),'--seed-b',str(sb),'--steps',str(steps),
             '--rounds','1','--warmup-ms','0','--verify','1']
        if records:cmd+=['--records',str(records)]
        module_hash=''
        if backend=='original':
            original=base/(f'd{d}-u{u}' if mode in modes[:3] else f'd{d}-u{u}-lds{s}')
            if original not in metadata:metadata[original]=(original/'metadata.txt').read_text()
            symbol=re.search(r'\.name:\s*(_Z12matmul_probeILi'+str(modes.index(mode))+r'ELi'+str(loads)+r'ELb1EE\S+)',metadata[original])[1]
            module=original/'probe.hsaco';module_hash=hashlib.sha256(module.read_bytes()).hexdigest()
            cmd+=['--module',str(module),'--symbol',symbol]
        commands.write(shlex.join(cmd)+'\n');commands.flush()
        if not (a.resume and out.exists() and log.exists() and 'COMPLETE verified=1' in log.read_text()):
            with out.open('w') as o,log.open('w') as e:result=subprocess.run(cmd,stdout=o,stderr=e,check=False)
            if result.returncode not in (0,2):raise RuntimeError(f'{tag}: execution error {result.returncode}; inspect {log}')
        data=list(csv.DictReader(out.open()));r=data[-1]
        summary.append(dict(backend=backend,depth=d,unroll=u,stride=s,mode=mode,loads=loads,wait=wait,
                            pattern=pattern,seed_a=sa,seed_b=sb,steps=steps,records=records,
                            mismatches=int(r['mismatch_count']),guard_errors=int(r['guard_errors']),
                            max_abs_error=float(r['max_abs_error']),valid=int(r['valid']),module_sha256=module_hash))
        if idx%25==0 or int(r['valid'])!=1:print(f'{idx+1}/{len(jobs)} {tag}: valid={r["valid"]} mismatches={r["mismatch_count"]}',flush=True)
with (a.output/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
correct=summary
failed=[r for r in correct if not r['valid']]
text=(f'Waited full-output cases: {len(correct)}, failed: {len(failed)}.\n'
      f'Original code-object cases: {sum(r["backend"]=="original" for r in correct)}.\n'
      f'All cases check 4096 outputs and 32 guard words. Mathematical equality is exact for these dyadic inputs.\n')
(a.output/'summary.txt').write_text(text)
with (a.output/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(text)
if failed:raise SystemExit(1)
