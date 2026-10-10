# Complete A/B chunk-order matrix, grids up to 104

User request: A/B × **104/80/64 blocks × 4/8/16 chunks × both orderings**.
This is a new matched sweep of all 36 cases, not a table assembled from
historical runs. No reader grids above 104, and no 32-block cases.

## Conditions

Native O3 HIP/C++20, gfx90a, HIP logical GPU 0 / PCI 0000:31:00.0,
104 physical CUs, CPU+memory NUMA node 0, no model/context. Block size 256.
Phase-only compiler scheduling constraints: RESEARCH_SCHED_BARRIERS=1,
RESEARCH_LOAD_BARRIERS=0. Same flat affine uint32_t input / HIP uint4,
with one 4 GiB allocation per process. No clock overrides; active telemetry
samples 1700/1600 MHz core/memory in both runs.

Two independent processes, nine jointly shuffled rounds each, twelve reads
per timed batch. Every case measures 1/2/4 GiB. The following are **pooled
medians over 18 batch measurements**, rounded to integer decimal useful-byte
GB/s. Each cell is **grid-spaced / adjacent**; the slash does NOT represent
the two processes. Individual process medians, ranges and raw rounds are
also retained. Pooling is descriptive, not an assertion of 18 independent
allocations or a hardware-counter isolation.

### 1 GiB

| Blocks | Chunks | A: grid / adjacent | B: grid / adjacent |
|---:|---:|---:|---:|
| 104 | 4 | 1449 / 1469 | 1380 / 1422 |
| 104 | 8 | 1353 / 1370 | 1341 / 1372 |
| 104 | 16 | 1340 / 1306 | 1339 / 1394 |
| 80 | 4 | 1452 / 1461 | 1360 / 1399 |
| 80 | 8 | 1377 / 1374 | 1284 / 1293 |
| 80 | 16 | 1361 / 1354 | 1265 / 1318 |
| 64 | 4 | 1334 / 1412 | 1218 / 1302 |
| 64 | 8 | 1438 / 1361 | 1291 / 1289 |
| 64 | 16 | 1369 / 1334 | 1234 / 1278 |

### 2 GiB

| Blocks | Chunks | A: grid / adjacent | B: grid / adjacent |
|---:|---:|---:|---:|
| 104 | 4 | 1409 / 1442 | 1344 / 1411 |
| 104 | 8 | 1347 / 1364 | 1366 / 1378 |
| 104 | 16 | 1336 / 1297 | 1372 / 1400 |
| 80 | 4 | 1460 / 1460 | 1338 / 1396 |
| 80 | 8 | 1371 / 1371 | 1295 / 1298 |
| 80 | 16 | 1359 / 1356 | 1313 / 1334 |
| 64 | 4 | 1337 / 1408 | 1209 / 1299 |
| 64 | 8 | 1415 / 1356 | 1294 / 1268 |
| 64 | 16 | 1364 / 1334 | 1240 / 1285 |

### 4 GiB

| Blocks | Chunks | A: grid / adjacent | B: grid / adjacent |
|---:|---:|---:|---:|
| 104 | 4 | 1398 / 1413 | 1334 / 1384 |
| 104 | 8 | 1338 / 1358 | 1377 / 1387 |
| 104 | 16 | 1326 / 1287 | 1387 / 1397 |
| 80 | 4 | 1411 / 1426 | 1321 / 1359 |
| 80 | 8 | 1367 / 1369 | 1319 / 1316 |
| 80 | 16 | 1360 / 1356 | 1353 / 1349 |
| 64 | 4 | 1338 / 1401 | 1207 / 1282 |
| 64 | 8 | 1383 / 1357 | 1287 / 1269 |
| 64 | 16 | 1364 / 1336 | 1235 / 1309 |

## Interpretation and validation

- A favors four chunks at 104/80; at 64 it favors eight grid-spaced chunks
  or four adjacent ones, depending on size and timing distribution.
- B benefits from 104 blocks; adjacent/four chunks leads at 1 GiB, while
  larger batches catch up at 4 GiB. At 80/64, more chunks are not a
  universal remedy. There is no single best chunk ordering/count for all
  these shapes and sizes.
- Complete issued batches remain verified before the first consumption
  wait: A 4/8/16 wide loads, B 16/32/64 scalar loads. Cache-policy flags
  and scratch traffic remain absent; register-lane/accumulator moves are
  recorded. Resource/codegen caveats from chunk-order.md still apply.
- The embedded GPU code object is byte-identical to order-tail-smoke:
  SHA-256 18392112ab88caa6f91668b46fb30c8907e3bd43f2d1d2439e85cb4b965e0df7.
  This extension changes the host candidate matrix, not GPU instructions.
- All 36 cases pass empty/partial-record/tile/group-boundary checks and
  every timed checksum. Each process has 108 summaries with nine samples;
  pooling yields 108 summaries with eighteen samples.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-order-matrix \
  default order-matrix 9 phase
```

Tracked: `results/order-matrix-{a,b,pooled}.csv`,
`order-matrix-resources.csv`, `order-matrix-loads.csv`.
Full snapshots, raw rounds, ISA and telemetry:
`temp/vram-coalescing/order-matrix-{a,b}/`.
`order-matrix-pooled/` is analysis only: concatenated raw observations with
second-process round IDs offset by nine, passed through the ordinary
summarizer. It is not a third timing run.
No baseline promotion, production changes or system modifications.
