#!/usr/bin/env python3
"""Equal-width native watermark sweep; all samples verified, prelaunch hashes."""
import argparse,csv,hashlib,json,random,re,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
from inspect_window import audit,loop_body
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=root/'temp/lds-staging/windows/template');p.add_argument('--nowait',type=Path,default=root/'temp/lds-staging/windows/nowait');p.add_argument('--artifacts',type=Path,default=root/'temp/lds-staging/windows/variants');p.add_argument('--output',type=Path,default=here/'results/windows');a=p.parse_args()
build=a.build.resolve();nowait=a.nowait.resolve();art=a.artifacts.resolve();out=a.output.resolve()
for folder in (out,art):
    if folder.exists() and any(folder.iterdir()):p.error('choose fresh result/artifact directories: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
for folder in (build,nowait):subprocess.run(['sha256sum','-c',str(folder/'hashes.txt')],check=True,stdout=subprocess.DEVNULL)
for sub in ('source','builds','raw'):(out/sub).mkdir()
for name in ('wait-bandwidth.hip','generate-window.py','build-window.sh','inspect_window.py','run_windows.py'):shutil.copy2(here/name,out/'source'/name)
for folder in (build,nowait):
    dst=out/'builds'/folder.name;dst.mkdir()
    for name in ('loads.hpp','compiler-flags.txt','hashes.txt','metadata.txt','loop.txt','inventory.json'):shutil.copy2(folder/name,dst/name)
thresholds=list(range(33))+[40,48,56,62,63]
mc=subprocess.check_output(['llvm-mc-21','-triple=amdgcn-amd-amdhsa','-mcpu=gfx90a','-show-encoding'],input='\n'.join(f's_waitcnt vmcnt({n})' for n in thresholds)+'\n',text=True)
(out/'wait-encodings.txt').write_text(mc)
enc={n:bytes(int(b,16) for b in m.split(',')) for n,m in zip(thresholds,re.findall(r'encoding: \[([^\]]+)\]',mc))};assert len(enc)==len(thresholds) and all(len(b)==4 for b in enc.values())
exe=(build/'probe').read_bytes();obj=(build/'probe.hsaco').read_bytes();assert exe.count(obj)==1;embedded=exe.index(obj)
sections=json.loads(subprocess.check_output(['llvm-readobj-21','--sections','--elf-output-style=JSON',str(build/'probe.hsaco')]))[0]['Sections']
text=next(s['Section'] for s in sections if s['Section']['Name']['Name']=='.text')
seq=loop_body((build/'probe.isa').read_text());sites=[seq[i+1] for i,s in enumerate(seq) if s['asm'].startswith('global_load')];assert len(sites)==64 and all(s['raw']==enc[63] for s in sites)
manifest={};variants={}
for n in thresholds:
    label=f'vm{n:02d}';dest=art/label;dest.mkdir();patched=bytearray(obj);allowed=set();patches=[]
    for site in sites:
        pos=site['pc']-text['Address']+text['Offset'];assert obj[pos:pos+4]==enc[63]
        patched[pos:pos+4]=enc[n];allowed.update(range(pos,pos+4));patches.append(dict(pc=hex(site['pc']),offset=pos,old=enc[63].hex(),new=enc[n].hex()))
    delta={i for i,(x,y) in enumerate(zip(obj,patched)) if x!=y};assert delta<=allowed
    image=exe[:embedded]+bytes(patched)+exe[embedded+len(obj):];assert len(image)==len(exe)
    program=dest/'probe';program.write_bytes(image);program.chmod((build/'probe').stat().st_mode);(dest/'probe.hsaco').write_bytes(patched)
    subprocess.run(['llvm-objdump-21','--offloading',str(program)],stdout=subprocess.DEVNULL,check=True)
    assert Path(str(program)+'.0.hipv4-amdgcn-amd-amdhsa--gfx90a').read_bytes()==bytes(patched)
    (dest/'probe.isa').write_text(subprocess.check_output(['llvm-objdump-21','-d','--demangle',str(dest/'probe.hsaco')],text=True));shutil.copy2(build/'metadata.txt',dest/'metadata.txt')
    audit(dest,str(n));newseq=loop_body((dest/'probe.isa').read_text());assert len(newseq)==len(seq)
    assert all(x==y for x,y in zip(seq,newseq) if x['pc'] not in {s['pc'] for s in sites})
    rec=out/'builds'/label;rec.mkdir()
    for name in ('loop.txt','inventory.json'):shutil.copy2(dest/name,rec/name)
    manifest[label]=dict(wait=n,peak_window_bound=None if n==63 else n+1,sha256=hashlib.sha256(image).hexdigest(),hsaco_sha256=hashlib.sha256(patched).hexdigest(),changed_bytes=len(delta),patches=patches,program=str(program));variants[label]=program
variants['none']=nowait/'probe';manifest['none']=dict(wait=None,peak_window_bound=None,sha256=hashlib.sha256((nowait/'probe').read_bytes()).hexdigest(),program=str(nowait/'probe'))
(out/'native-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[];launch_hashes=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    jobs=[(name,n) for name in variants for n in (1,8)];random.Random(1729).shuffle(jobs)
    for repeat in (1,2):
        for name,ctas in jobs if repeat==1 else reversed(jobs):
            program=variants[name];sha=hashlib.sha256(program.read_bytes()).hexdigest();assert sha==manifest[name]['sha256'];launch_hashes.append(dict(variant=name,ctas=ctas,repeat=repeat,sha256=sha))
            cmd=['numactl','--cpunodebind=0','--membind=0',str(program),'--ctas',str(ctas),'--groups','16384','--rounds','9','--warmup-ms','200'];commands.write(shlex.join(cmd)+'\n');commands.flush();stem=f'{name}-n{ctas}-r{repeat}'
            with (out/'raw'/(stem+'.csv')).open('w') as f,(out/'raw'/(stem+'.log')).open('w') as e:subprocess.run(cmd,stdout=f,stderr=e,check=True,timeout=90)
            with (out/'raw'/(stem+'.csv')).open() as f:data=list(csv.DictReader(f))
            assert len(data)==9*ctas and 'COMPLETE all per-CTA LDS words and output guards exact' in (out/'raw'/(stem+'.log')).read_text()
            per_round=[];bandwidth=[];overlaps=[]
            for r in range(9):
                samples=[s for s in data if int(s['round'])==r];assert {int(s['cta']) for s in samples}==set(range(ctas))
                for s in samples:
                    assert s['valid']=='1' and s['mismatches']==s['guards']=='0' and int(s['ctas'])==ctas
                    span=max(int(s[f'issue{w}']) for w in range(4))-min(int(s[f'begin{w}']) for w in range(4));assert span==int(s['issue_cycles'])
                    assert int(s['loads_per_wave'])==1048576 and int(s['requested_bytes_per_cta'])==1073741824
                    assert abs(float(s['cycles_per_four_wave_load'])-span/1048576)<1e-8
                per_round.append(median(float(s['cycles_per_four_wave_load']) for s in samples))
                starts=[min(int(s[f'wall_begin{w}']) for w in range(4)) for s in samples];ends=[max(int(s[f'wall_issue{w}']) for w in range(4)) for s in samples]
                overlaps.append(max(starts)<min(ends))
                localticks=ends[0]-starts[0];tick_us=float(samples[0]['issue_us'])/localticks;grid_us=(max(ends)-min(starts))*tick_us
                bandwidth.append(ctas*1073741824/(grid_us*1000))
            assert all(overlaps),'CTA body intervals did not overlap'
            row=dict(variant=name,wait=manifest[name]['wait'],peak_window_bound=manifest[name]['peak_window_bound'],ctas=ctas,repeat=repeat,samples=9,checked_cta_rows=len(data),cycles_per_four_wave_load=median(per_round),min_round=min(per_round),max_round=max(per_round),per_cta_requested_B_per_cycle=1024/median(per_round),aggregate_requested_GBps=median(bandwidth),core_MHz=median(float(s['core_MHz']) for s in data),all_ctas_overlap=True)
            rows.append(row)
            with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
            print(f"{stem}: {row['cycles_per_four_wave_load']:.4f} cycles/load, {row['aggregate_requested_GBps']:.2f} GB/s",flush=True)
(out/'launch-hashes.json').write_text(json.dumps(launch_hashes,indent=2)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print(f"COMPLETE {len(rows)*9} checked grid timings, {sum(r['checked_cta_rows'] for r in rows)} checked CTA records; all LDs/guards exact.")
