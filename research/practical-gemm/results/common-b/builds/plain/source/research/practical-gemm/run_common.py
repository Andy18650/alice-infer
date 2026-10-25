#!/usr/bin/env python3
"""Four shared B formats, 70B TP4/8 per-rank shapes, no runtime B repacking."""
import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from run import digest,check_build

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FORMATS=('native','plain','block128','block256')


def cases():
    checks=[]
    ns=(1,31,33,127,128,129,255,256,257,383,384,511,512,513,769)
    ms=(1,3,17,31,32,33,63,64,65,127,128,129,257)
    for stage in range(30):
        checks.append(dict(case=f'stages-{stage}',path='gemm',m=ms[stage%len(ms)],n=ns[stage%len(ns)],k=32*stage,scale=stage%4,finite=0,tp=0))
    for m in (1,17,129):
        for k in (32,64):checks.append(dict(case=f'finite-{m}-{k}',path='gemm',m=m,n=259,k=k,scale=2,finite=1,tp=0))
    for k in (0,32,128,160,352,65536):
        checks.append(dict(case=f'decode-k{k}',path='decode',m=1,n=257,k=k,scale=3,finite=0,tp=0))
    timing=[]
    for tp in (4,8):
        shapes={'qkv':(10240//tp,8192),'o':(8192,8192//tp),
                'gateup':(57344//tp,8192),'down':(8192,28672//tp)}
        for name,(n,k) in shapes.items():
            timing.append(dict(case=f'70b-tp{tp}-{name}-decode',path='decode',m=1,n=n,k=k,scale=2,finite=0,tp=tp))
            for m in (16,128,512,2048):
                timing.append(dict(case=f'70b-tp{tp}-{name}-m{m}',path='gemm',m=m,n=n,k=k,scale=2,finite=0,tp=tp))
    for m in (1664,3328,4992):
        for k in (8192,32768):timing.append(dict(case=f'control-{m//128*8}-k{k}',path='gemm',m=m,n=1024,k=k,scale=2,finite=0,tp=0))
    timing.append(dict(case='large-grid',path='gemm',m=4096,n=6656,k=32768,scale=2,finite=0,tp=0))
    return checks,timing


def analyze(output):
    rows=list(csv.DictReader((output/'samples.csv').open()));groups={}
    for row in rows:
        if row['kind']=='timing':groups.setdefault((row['case'],row['format'],row['repeat']),[]).append(row)
    summary=[]
    for (case,fmt,repeat),samples in groups.items():
        first=samples[0]
        item={key:first[key] for key in ('case','format','path','repeat','tp','m','n','k','ctas')}
        for key in ('event_us','useful_TFLOPs','useful_weight_GBs','requested_weight_GBs','median_core_MHz'):
            item[key]=statistics.median(float(s[key]) for s in samples)
        summary.append(item)
    with (output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
    heading=['# Shared-B format comparison','',
        'Single gfx90a GCD HIP0/BDF0000:31:00.0; CPU+memory NUMA0, automatic clocks, no model/context.',
        '70B cases are TP4/8 per-rank projection shapes, NOT full-model/collective throughput.',
        'Row-major device FP16 A, finite E4M3FN B, logical FP32 K128/N128 scales.',
        'No A/B repacking or FP32 conversion in either path. B is packed once outside timing.',
        'Two paired seeds/order-reversed repeats; means of nine-sample process medians.',
        'GEMM event covers the complete kernel; decode event covers partial+ordered scale/combine.',
        'GB/s below is logical N*K bytes / complete decode event, NOT measured physical HBM traffic.','']
    for path,title,metric in [('decode','Decode: useful weight GB/s','useful_weight_GBs'),('gemm','GEMM: useful TFLOP/s','useful_TFLOPs')]:
        lines=heading+[f'## {title}','',
            '|Case|M|N|K|native|plain|block128|block256|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for case in dict.fromkeys(r['case'] for r in summary if r['path']==path):
            samples=[r for r in summary if r['case']==case];first=samples[0]
            values=[]
            for fmt in FORMATS:
                pair=[r[metric] for r in samples if r['format']==fmt];assert len(pair)==2
                values.append(f'{statistics.mean(pair):.3f}')
            lines.append(f"|{case}|{first['m']}|{first['n']}|{first['k']}|"+'|'.join(values)+'|')
        (output/f'{path}.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if a.analyze_only:analyze(a.output);return
    if a.output.exists():raise RuntimeError('refusing existing result path')
    if not a.build_root:p.error('--build-root required')
    builds={f:(a.build_root/f).resolve() for f in FORMATS}
    for build in builds.values():check_build(build)
    a.output.mkdir(parents=True)
    provenance=dict(argv=sys.argv,driver_sha256=digest(Path(__file__)),placement='numactl --cpunodebind=0 --membind=0',
        clock_policy='automatic; no overrides',model_context=None,scope='70B TP4/8 per-rank synthetic projections; no collectives',builds={},processes=[])
    for fmt,build in builds.items():
        target=a.output/'builds'/fmt;target.mkdir(parents=True)
        for name in ('compute.hpp','kernel.hpp','probe.isa','metadata.txt','compiler-flags.txt','compiler-version.txt','dynamic-dependencies.txt','hashes.txt'):
            shutil.copy2(build/name,target/name)
        for line in (build/'hashes.txt').read_text().splitlines():
            _,name=line.split('  ',1);path=Path(name)
            if path.is_relative_to(ROOT/'research'):
                dest=target/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        for name in ('run_common.py','audit_common.py','verify_common.py','run.py','verify.py'):
            shutil.copy2(HERE/name,target/'source'/'research'/'practical-gemm'/name)
        provenance['builds'][fmt]=dict(directory=str(build),executable_sha256=digest(build/'probe'),code_object_sha256=digest(build/'probe.hsaco'))
    with (a.output/'checks.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'audit_common.py'),'--build-root',str(a.build_root),'--output',str(a.output/'checks')],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    checks,timing=cases();rows=[]
    for kind,repeat,selection in [('verification',0,checks),('timing',1,timing),('timing',2,list(reversed(timing)))]:
        order=FORMATS if repeat!=2 else tuple(reversed(FORMATS))
        for case in selection:
            for fmt in order:
                build=builds[fmt];check_build(build)
                target=a.output/kind/f'r{repeat}'/case['case']/fmt;target.mkdir(parents=True)
                command=['numactl','--cpunodebind=0','--membind=0',str(build/'probe'),
                    '--path',case['path'],'--m',str(case['m']),'--n',str(case['n']),'--k',str(case['k']),
                    '--scale-pattern',str(case['scale']),'--b-pattern',str(case['finite']),
                    '--seed-a',str(1729+repeat),'--seed-b',str(2718+repeat),
                    '--rounds','9' if kind=='timing' else '1','--warmup-ms','200' if kind=='timing' else '0',
                    '--stamps-prefix',str(target/'clocks')]
                (target/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                with (target/'stdout.csv').open('w') as out,(target/'stderr.txt').open('w') as err:
                    subprocess.run(command,check=True,stdout=out,stderr=err)
                check_build(build)
                samples=list(csv.DictReader((target/'stdout.csv').open()))
                assert len(samples)==(9 if kind=='timing' else 1)
                assert all(s['valid']=='1' and (s['path']!='gemm' or s['resident_ctas_per_cu']=='2') for s in samples)
                for s in samples:s.update(kind=kind,repeat=repeat,case=case['case'],tp=case['tp'])
                rows.extend(samples)
                provenance['processes'].append(dict(case=case['case'],kind=kind,repeat=repeat,format=fmt,path=case['path'],tp=case['tp'],
                    executable_sha256=digest(build/'probe'),checked_timed_invocations=len(samples),checked_interop_invocations=2))
                print(kind,repeat,case['case'],fmt,'PASS',flush=True)
    with (a.output/'samples.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    analyze(a.output)
    for fmt,build in builds.items():
        rejected=subprocess.run([str(build/'probe'),'--k','31'],capture_output=True,text=True)
        assert rejected.returncode!=0 and 'multiple-of32' in rejected.stderr
        (a.output/'checks'/f'{fmt}-unsupported-k.txt').write_text(rejected.stderr)
    before=digest(builds['native']/'probe')
    rejected=subprocess.run(['bash',str(HERE/'build_common.sh'),'native',str(builds['native'])],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'refusing nonempty' in rejected.stderr and before==digest(builds['native']/'probe')
    (a.output/'checks'/'nonempty-build.txt').write_text(rejected.stderr)
    provenance.update(checked_timed_invocations=len(rows),checked_interop_invocations=2*len(provenance['processes']),process_count=len(provenance['processes']))
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    from verify_common import verify_results
    path=a.output/'checks'/'summary.json';summary=json.loads(path.read_text())
    summary['captured_result_checks']=verify_results(a.output,check_manifest=False)
    path.write_text(json.dumps(summary,indent=2)+'\n')
    manifest={str(path.relative_to(a.output)):digest(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE',len(rows),'timed checked matmuls;',provenance['checked_interop_invocations'],'interop checked matmuls;',len(provenance['processes']),'hashed processes')


if __name__=='__main__':main()
