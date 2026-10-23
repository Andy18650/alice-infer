# Native GEMM matched results

Native HIP O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, no model/context/clock overrides. CPU packing/H2D/reference excluded; full kernel HIP event used for TFLOP/s. Two counterbalanced process repeats, nine samples each.

| M x N x K | Tile | Useful TFLOP/s run1/run2 | Effective core MHz | Global clock-adjusted efficiency |
|---|---:|---:|---:|---:|
|1664x1024x8192|128|113.15/113.23 | 1475.46/1474.40 | 72.01/72.12%|
|1664x1024x8192|256|34.61/34.44 | 1632.42/1631.37 | 19.90/19.80%|
|3328x2048x8192|128|137.69/137.52 | 1427.69/1428.08 | 90.57/90.44%|
|3328x2048x8192|256|122.88/123.44 | 1486.56/1487.97 | 77.54/77.93%|
|4096x6656x8192|128|141.52/142.13 | 1425.15/1424.85 | 93.28/93.65%|
|4096x6656x8192|256|124.66/124.88 | 1486.57/1486.57 | 78.74/78.89%|
|4096x6656x32768|128|144.07/144.02 | 1417.94/1418.13 | 95.42/95.38%|
|4096x6656x32768|256|128.36/128.37 | 1482.96/1483.19 | 81.28/81.27%|

Observed maximum simultaneous CTA-body intervals in the representative long-K launch: tile128=312, tile256=104. This is body-overlap evidence, not exact allocation timestamps or a CTA-to-CU map.

Checked launches: 180. Large cases use sampled exact references plus all-C finite/poison and guard scans; do not relabel them full references.
