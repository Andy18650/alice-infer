#!/usr/bin/env python3
"""Performance-only, equal-width instruction patches; NEVER numerical validation."""
import argparse,csv,hashlib,json,random,re,shlex,shutil,subprocess
from pathlib import Path
from statistics import median
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser()
p.add_argument('--build',type=Path,default=root/'temp/fp16-gemm64/active')
p.add_argument('--artifacts',type=Path,default=root/'temp/fp16-gemm64/ablations')
p.add_argument('--output',type=Path,default=here/'results/ablations')
a=p.parse_args();build=a.build.resolve();art=a.artifacts.resolve();out=a.output.resolve()
for folder in (art,out):
    if folder.exists() and any(folder.iterdir()):p.error('refusing nonempty artifact/result directory: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
subprocess.run(['sha256sum','-c',str(build/'hashes.txt')],stdout=subprocess.DEVNULL,check=True)
exe=(build/'probe').read_bytes();obj=(build/'probe.hsaco').read_bytes()
assert exe.count(obj)==1,'expected one uncompressed embedded code object'
embedded=exe.index(obj)
sections=json.loads(subprocess.check_output(['llvm-readobj-21','--sections','--elf-output-style=JSON',str(build/'probe.hsaco')]))[0]['Sections']
text=next(x['Section'] for x in sections if x['Section']['Name']['Name']=='.text')
isa=(build/'probe.isa').read_text();instructions=[]
for section in re.split(r'(?=^0[0-9a-f]+ <)',isa,flags=re.M):
    if '<gemm_k8(' not in (section.splitlines()[0] if section else ''):continue
    for line in section.splitlines():
        m=re.search(r'// ([0-9a-fA-F]+):((?: [0-9a-fA-F]{8})+)',line)
        if not m:continue
        instructions.append(dict(pc=int(m[1],16),asm=line.split('//')[0].strip(),
                                 raw=b''.join(int(w,16).to_bytes(4,'little') for w in m[2].split())))
assert instructions
bulk=None
for i,ins in enumerate(instructions):
    if not ins['asm'].startswith('s_cbranch_scc1 '):continue
    d=int(ins['asm'].split()[1],0);d=d-65536 if d>=32768 else d
    if d>=0:continue
    target=ins['pc']+4+4*d
    seq=[x for x in instructions[:i+1] if x['pc']>=target]
    if sum(x['asm'].startswith('v_mfma_') for x in seq)==16:bulk=seq
assert bulk is not None
(out/'bulk-baseline.txt').write_text('\n'.join(f"{x['pc']:08x}: {x['asm']}" for x in bulk)+'\n')
(out/'elf-sections.json').write_text(json.dumps(sections,indent=2)+'\n')
assembly=['s_nop 0']+[f'v_pk_fma_f32 v[{v}:{v+1}], v[{v}:{v+1}], v[{v+2}:{v+3}], v[{v}:{v+1}]' for v in (0,4)]
mc=subprocess.check_output(['llvm-mc-21','-triple=amdgcn-amd-amdhsa','-mcpu=gfx90a','-show-encoding'],input='\n'.join(assembly)+'\n',text=True)
(out/'replacement-encodings.txt').write_text(mc)
enc=[bytes(int(b,16) for b in m.split(',')) for m in re.findall(r'encoding: \[([^\]]+)\]',mc)]
assert [len(x) for x in enc]==[4,8,8]
variants={};manifest={}
for variant in ('baseline','mfma-to-pkfma','vmcnt-to-nop','barrier-to-nop'):
    patched=bytearray(obj);patches=[];allowed=set()
    for ins in bulk:
        op=ins['asm'];replacement=None;spelling=''
        if variant=='mfma-to-pkfma' and op.startswith('v_mfma_'):
            v=int(re.search(r'v\[(\d+):',op)[1]);assert v in (0,4)
            replacement=enc[1+v//4];spelling=assembly[1+v//4]
        elif variant=='vmcnt-to-nop' and op=='s_waitcnt vmcnt(24)':replacement=enc[0];spelling=assembly[0]
        elif variant=='barrier-to-nop' and op=='s_barrier':replacement=enc[0];spelling=assembly[0]
        if replacement is None:continue
        pos=ins['pc']-text['Address']+text['Offset']
        assert len(replacement)==len(ins['raw'])
        assert obj[pos:pos+len(replacement)]==ins['raw']
        patched[pos:pos+len(replacement)]=replacement;allowed.update(range(pos,pos+len(replacement)))
        patches.append(dict(gpu_pc=hex(ins['pc']),code_object_offset=pos,executable_offset=embedded+pos,
                            original=op,replacement=spelling,old_hex=ins['raw'].hex(),new_hex=replacement.hex()))
    assert len(patches)==(0 if variant=='baseline' else 16)
    delta={i for i,(x,y) in enumerate(zip(obj,patched)) if x!=y}
    assert delta<=allowed and (not delta if variant=='baseline' else bool(delta))
    image=exe[:embedded]+bytes(patched)+exe[embedded+len(obj):]
    assert len(image)==len(exe) and image[:embedded]==exe[:embedded] and image[embedded+len(obj):]==exe[embedded+len(obj):]
    dest=art/variant;dest.mkdir();program=dest/'probe';program.write_bytes(image);program.chmod((build/'probe').stat().st_mode)
    (dest/'probe.hsaco').write_bytes(patched)
    dis=subprocess.check_output(['llvm-objdump-21','-d','--demangle',str(dest/'probe.hsaco')],text=True)
    (dest/'probe.isa').write_text(dis)
    folder=out/'builds'/variant;folder.mkdir(parents=True)
    # A second extraction from the executable proves the embedded object was patched too.
    subprocess.run(['llvm-objdump-21','--offloading',str(program)],stdout=subprocess.DEVNULL,check=True)
    extracted=Path(str(program)+'.0.hipv4-amdgcn-amd-amdhsa--gfx90a').read_bytes()
    assert extracted==bytes(patched)
    view=[]
    for line in dis.splitlines():
        m=re.search(r'// ([0-9a-fA-F]+):',line)
        if m and bulk[0]['pc']<=int(m[1],16)<=bulk[-1]['pc']:view.append(line)
    (folder/'bulk.txt').write_text('\n'.join(view)+'\n')
    assert len(view)==len(bulk)
    assert sum('v_mfma_' in x for x in view)==(0 if variant=='mfma-to-pkfma' else 16)
    assert sum('v_pk_fma_' in x for x in view)==(16 if variant=='mfma-to-pkfma' else 0)
    assert sum('vmcnt(24)' in x for x in view)==(0 if variant=='vmcnt-to-nop' else 16)
    assert sum('s_barrier' in x for x in view)==(0 if variant=='barrier-to-nop' else 16)
    manifest[variant]=dict(program=str(program),executable_sha256=hashlib.sha256(image).hexdigest(),
                           hsaco_sha256=hashlib.sha256(patched).hexdigest(),changed_bytes=len(delta),patches=patches)
    variants[variant]=program
(out/'patches.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ('hashes.txt','compiler-flags.txt','inventory.json','compute.hpp'):shutil.copy2(build/name,out/name)
for name in ('gemm64.hip','generate.py','data.hpp','build.sh'):
    (out/'source').mkdir(exist_ok=True);shutil.copy2(here/name,out/'source'/name)
shutil.copy2(Path(__file__),out/'source/ablate.py')
(out/'timing').mkdir()
# Verify the original baseline once, outside the performance-only comparisons.
verify=['numactl','--cpunodebind=0','--membind=0',str(variants['baseline']),'--k','32768','--rounds','1','--warmup-ms','0']
with (out/'baseline-check.csv').open('w') as f,(out/'baseline-check.log').open('w') as e:subprocess.run(verify,stdout=f,stderr=e,check=True)
with (out/'baseline-check.csv').open() as f:assert all(x['valid']=='1' for x in csv.DictReader(f))
rows=[]
with (out/'amd-smi-before.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
with (out/'commands.txt').open('w') as commands:
    commands.write('# verified baseline only\n'+shlex.join(verify)+'\n')
    for repeat in (1,2):
        order=list(variants);random.Random(1729+repeat).shuffle(order)
        for variant in order:
            cmd=['numactl','--cpunodebind=0','--membind=0',str(variants[variant]),'--k','32768',
                 '--verify','0','--rounds','9','--warmup-ms','200']
            commands.write('# PERFORMANCE ONLY; NO NUMERICAL CLAIM\n'+shlex.join(cmd)+'\n');commands.flush()
            stem=f'{variant}-run{repeat}'
            with (out/'timing'/(stem+'.csv')).open('w') as f,(out/'timing'/(stem+'.log')).open('w') as e:
                subprocess.run(cmd,stdout=f,stderr=e,check=True,timeout=60)
            with (out/'timing'/(stem+'.csv')).open() as f:samples=list(csv.DictReader(f))
            assert len(samples)==9 and all(x['valid']=='-1' and x['steps']=='4096' for x in samples)
            for x in samples:
                span=max(int(x[f'end{w}']) for w in range(4))-min(int(x[f'begin{w}']) for w in range(4))
                assert span==int(x['cta_cycles']) and abs(float(x['cycles_per_mfma'])-span/4096)<1e-8
            row=dict(variant=variant,repeat=repeat,samples=9,verification='DISABLED; performance only',
                     cycles_per_mfma=median(float(x['cycles_per_mfma']) for x in samples),
                     body_us=median(float(x['body_us']) for x in samples),
                     launch_us=median(float(x['launch_us']) for x in samples),
                     core_MHz=median(float(x['core_MHz']) for x in samples))
            rows.append(row);print(row,flush=True)
with (out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
text=['PERFORMANCE-ONLY ABLATIONS: numerical verification disabled in all timing runs.',
      'Native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768, no model/context/clock overrides.',
      'Two randomized process-order repeats, nine samples each, selected-kernel warmup200ms.',
      'Only16 steady-state sites patched; startup and32-record tail unchanged. No additive cost decomposition.',
      'variant | cycles/logical four-wave K8 step run1/run2 | whole body us run1/run2 | core MHz run1/run2']
for variant in variants:
    r=[x for x in rows if x['variant']==variant]
    text.append(variant+' | '+' | '.join('/'.join(f'{x[k]:.3f}' for x in r) for k in ('cycles_per_mfma','body_us','core_MHz')))
(out/'summary.txt').write_text('\n'.join(text)+'\n')
with (out/'amd-smi-after.txt').open('w') as f:subprocess.run(['amd-smi'],stdout=f,stderr=subprocess.STDOUT,check=False)
print('\n'.join(text))
