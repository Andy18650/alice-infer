# Native GEMM matched results

Native HIP O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, no model/context/clock overrides. CPU packing/H2D/reference excluded; full kernel HIP event used for TFLOP/s. Two counterbalanced process repeats, nine samples each.

| M x N x K | Tile | Useful TFLOP/s run1/run2 | Effective core MHz | Global clock-adjusted efficiency |
|---|---:|---:|---:|---:|
|1664x1024x8192|128|113.15/113.23 | 1474.39/1476.56 | 72.06/72.03%|
|1664x1024x8192|256|34.46/34.41 | 1630.54/1631.15 | 19.85/19.82%|
|3328x2048x8192|128|137.66/137.66 | 1427.40/1428.14 | 90.55/90.51%|
|3328x2048x8192|256|122.66/122.68 | 1485.93/1486.82 | 77.47/77.50%|
|4096x6656x8192|128|141.58/142.07 | 1423.91/1423.83 | 93.49/93.66%|
|4096x6656x8192|256|124.59/124.92 | 1486.01/1487.31 | 78.74/78.86%|
|4096x6656x32768|128|144.07/144.17 | 1417.51/1417.89 | 95.44/95.47%|
|4096x6656x32768|256|128.37/128.35 | 1482.71/1482.74 | 81.29/81.29%|

Observed maximum simultaneous CTA-body intervals in the representative long-K launch: tile128=312, tile256=104. This is body-overlap evidence, not exact allocation timestamps or a CTA-to-CU map.

Checked launches: 180. Large cases use sampled exact references plus all-C finite/poison and guard scans; do not relabel them full references.
