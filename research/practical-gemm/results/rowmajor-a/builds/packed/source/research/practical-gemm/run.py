#!/usr/bin/env python3
"""Counterbalanced row-major-A vs packed-A study; no tuning or vendor GEMM."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_build(build):
    for line in (build/'hashes.txt').read_text().splitlines():
        expected,path=line.split('  ',1)
        assert digest(Path(path))==expected,('changed build/source',path)


def cases():
    boundaries=[1,3,16,17,31,32,33,63,64,65,127,128,129]
    checks=[]
    for stages in range(30):
        checks.append((f'stages-{stages}',boundaries[stages%len(boundaries)],131,32*stages,stages%4,0))
    for m in boundaries:
        checks.append((f'm-{m}',m,131,352,2,0))
    checks.extend([('long-k',1,129,65536,3,0),('three-row-tiles',257,257,64,2,0)])
    for m in (1,17,129):
        for k in (32,64):checks.append((f'finite-{m}-{k}',m,131,k,2,1))
    for n in (1,31,32,33,127,128,129,257):
        checks.append((f'n-{n}',33,n,160,1,0))
    timing=[]
    for name,n,k in [('8b-qkv',6144,4096),('8b-gateup',28672,4096),('8b-down',4096,14336)]:
        for m in (1,16,128,512,2048):timing.append((f'{name}-m{m}',m,n,k,2,0))
    for m in (33,129):timing.append((f'8b-qkv-m{m}',m,6144,4096,2,0))
    for m in (1664,3328,4992):
        for k in (8192,32768):timing.append((f'control-{m//128*8}-{k}',m,1024,k,2,0))
    timing.append(('large-grid',4096,6656,32768,2,0))
    return checks,timing


def analyze(output):
    rows=list(csv.DictReader((output/'samples.csv').open()))
    groups={}
    for r in rows:
        if r['kind']=='timing':
            groups.setdefault((r['case'],r['layout'],r['protocol'],int(r['repeat'])),[]).append(r)
    process=[]
    for (case,layout,protocol,repeat),samples in groups.items():
        first=samples[0]
        process.append(dict(case=case,layout=layout,protocol=protocol,repeat=repeat,
            m=first['m'],n=first['n'],k=first['k'],ctas=first['ctas'],
            event_us=statistics.median(float(r['event_us']) for r in samples),
            pack_us=statistics.median(float(r['pack_us']) for r in samples),
            useful_TFLOPs=statistics.median(float(r['useful_TFLOPs']) for r in samples)))
    if process:
        with (output/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(process[0]));w.writeheader();w.writerows(process)
    lines=['# Row-major A: practical FP8-B shape comparison','',
        'CPU+memory NUMA0, HIP0/BDF0000:31:00.0, automatic clocks; no model/context.',
        'Times are means of two nine-sample process medians, microseconds.',
        'Kernel excludes A packing; activation includes the plain GPU pack+GEMM for packed A.',
        'Both start with device-resident row-major FP16 A; FP32 conversion/H2D/weight packing excluded.',
        'Negative delta means row-major is faster. No claim that the plain packer is optimal.','',
        '|Case|CTAs|Packed kernel us|Row-major kernel us|Kernel delta|Packed activation us|Row-major activation us|Activation delta|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in dict.fromkeys(r['case'] for r in process):
        values={}
        for layout in ('packed','rowmajor'):
            for protocol in ('kernel','activation'):
                selected=[r['event_us'] for r in process if r['case']==case and r['layout']==layout and r['protocol']==protocol]
                assert len(selected)==2,(case,layout,protocol)
                values[layout,protocol]=statistics.mean(selected)
        pk,rk,pa,ra=[values[key] for key in [('packed','kernel'),('rowmajor','kernel'),('packed','activation'),('rowmajor','activation')]]
        ctas=next(r['ctas'] for r in process if r['case']==case)
        lines.append(f'|{case}|{ctas}|{pk:.3f}|{rk:.3f}|{(rk/pk-1)*100:+.2f}%|{pa:.3f}|{ra:.3f}|{(ra/pa-1)*100:+.2f}%|')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return process


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed',type=Path);p.add_argument('--rowmajor',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    a=p.parse_args()
    if a.analyze_only:analyze(a.output);return
    if a.output.exists():raise RuntimeError('refusing existing result path')
    if not a.packed or not a.rowmajor:p.error('both builds required')
    builds={'packed':a.packed.resolve(),'rowmajor':a.rowmajor.resolve()}
    for build in builds.values():check_build(build)
    a.output.mkdir(parents=True)
    provenance={'argv':sys.argv,'driver_sha256':digest(Path(__file__)),
        'placement':'numactl --cpunodebind=0 --membind=0','model_context':None,
        'clock_policy':'automatic; no overrides','builds':{},'processes':[]}
    for layout,build in builds.items():
        target=a.output/'builds'/layout;target.mkdir(parents=True)
        for name in ('compute.hpp','probe.isa','metadata.txt','compiler-flags.txt','hashes.txt'):
            shutil.copy2(build/name,target/name)
        sources=[]
        for line in (build/'hashes.txt').read_text().splitlines():
            _,name=line.split('  ',1);path=Path(name)
            if path.is_relative_to(ROOT/'research'):
                destination=target/'source'/path.relative_to(ROOT)
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
                sources.append(str(path.relative_to(ROOT)))
        # Capture as-run driver/auditor too, without pretending they were compilation inputs.
        for name in ('run.py','audit.py','verify.py'):
            destination=target/'source'/'research'/'practical-gemm'/name
            shutil.copy2(HERE/name,destination)
        provenance['builds'][layout]={'directory':str(build),'executable_sha256':digest(build/'probe'),
            'code_object_sha256':digest(build/'probe.hsaco'),'compilation_sources':sources}
    subprocess.run([sys.executable,str(HERE/'audit.py'),'--packed',str(builds['packed']),
        '--rowmajor',str(builds['rowmajor']),'--output',str(a.output/'checks')],check=True,
        stdout=(a.output/'checks.log').open('w'),stderr=subprocess.STDOUT)
    checks,timing=cases();all_rows=[]
    for kind,repeat,selection in [('verification',0,checks),('timing',1,timing),('timing',2,list(reversed(timing)))]:
        order=['packed','rowmajor'] if repeat!=2 else ['rowmajor','packed']
        for case,m,n,k,scale,finite in selection:
            for layout in order:
                build=builds[layout];check_build(build)
                target=a.output/kind/f'r{repeat}'/case/layout;target.mkdir(parents=True)
                command=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),
                    '--m',str(m),'--n',str(n),'--k',str(k),'--scale-pattern',str(scale),'--b-pattern',str(finite),
                    '--seed-a',str(1729+repeat),'--seed-b',str(2718+repeat),
                    '--rounds','9' if kind=='timing' else '1','--warmup-ms','200' if kind=='timing' else '0',
                    '--stamps-prefix',str(target/'clocks')]
                (target/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                with (target/'stdout.csv').open('w') as stdout,(target/'stderr.txt').open('w') as stderr:
                    subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
                check_build(build)
                samples=list(csv.DictReader((target/'stdout.csv').open()))
                assert len(samples)==(18 if kind=='timing' else 2)
                assert all(int(r['valid'])==1 and int(r['resident_ctas_per_cu'])==2 for r in samples)
                for r in samples:r.update(kind=kind,repeat=repeat,case=case)
                all_rows.extend(samples)
                provenance['processes'].append({'kind':kind,'repeat':repeat,'case':case,'layout':layout,
                    'executable_sha256':digest(build/'probe'),'checked_launches':len(samples)})
                print(kind,repeat,case,layout,'PASS',flush=True)
    with (a.output/'samples.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    analyze(a.output)
    # Explicit rejection gates: unsupported K and nonempty measured build path.
    for layout,build in builds.items():
        result=subprocess.run([str(build/'probe'),'--k','31'],capture_output=True,text=True)
        assert result.returncode!=0 and 'multiple-of32' in result.stderr
        (a.output/'checks'/f'{layout}-unsupported-k.txt').write_text(result.stderr)
    before=digest(builds['rowmajor']/'probe')
    result=subprocess.run(['bash',str(HERE/'build.sh'),'rowmajor',str(builds['rowmajor'])],capture_output=True,text=True)
    assert result.returncode!=0 and 'refusing nonempty' in result.stderr and before==digest(builds['rowmajor']/'probe')
    (a.output/'checks'/'nonempty-build.txt').write_text(result.stderr)
    provenance.update(checked_launches=len(all_rows),process_count=len(provenance['processes']))
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    from verify import verify_results
    summary_path=a.output/'checks'/'summary.json'
    checks_summary=json.loads(summary_path.read_text())
    checks_summary['captured_result_checks']=verify_results(a.output,check_manifest=False)
    summary_path.write_text(json.dumps(checks_summary,indent=2)+'\n')
    manifest={str(path.relative_to(a.output)):digest(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',len(all_rows),'checked launches;',len(provenance['processes']),'hashed processes')


if __name__=='__main__':main()
