#!/usr/bin/env python3
"""Rebase archived source paths; verify preserved original native binaries."""
import hashlib
from pathlib import Path
base=Path(__file__).resolve().parent/'results'
for group in ('initial','swizzle','late','early','stage64','resident','final'):
 root=base/group
 files=list(root.glob('t*/hashes.txt')) if group!='final' else list((root/'builds').glob('*/hashes.txt'))
 assert files,group
 for manifest in files:
  for line in manifest.read_text().splitlines():
   sha,name=line.split(maxsplit=1);p=Path(name)
   if p.name in ('probe.hip','data.hpp','generate.py'):p=root/'source'/p.name
   elif p.name=='compute.hpp':p=manifest.parent/'compute.hpp'
   assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,(group,p)
 print(group,'source/header/original executable/code-object hashes pass')
for file in (base/'final').glob('ldd-*.txt'):
 text=file.read_text().lower();assert not any(s in text for s in ('rocblas','hipblas','libck'))
print('Recorded dynamic dependencies contain no vendor GEMM libraries')
