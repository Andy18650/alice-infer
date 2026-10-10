# Historical ceiling replay after the reported power cycle

Progress checkpoint: `521c46c`.
Explicit revert: `48db8d6`, restoring the research source state at `c104c08`.
No destructive reset; recent experiments remain in git history.

Pre-revert archive:
`temp/vram-checkpoints/20261010T084450Z-pre-revert/`
contains live sources, ring/blocks executable snapshots and assembly, raw
ring/blocks reports, the historical executable/source, and asset hashes.
The current ring assembly is `ring-c/disassembly.txt` inside that archive.

The restored `research/vram-ceiling/ceiling.hip` exactly matches the original
`temp/vram-ceiling/final-1/ceiling.hip`. Fresh rebuilding with the old build
script yields a byte-identical executable:

`c55b11add8a7cd0df89fbb4965be17122341d91de41eb4ccb8e7953be7fdccd1`

Each test executes the unchanged full historical sweep harness, not a new
module-only driver. The target is global uint2 (8 B/lane), unroll 16,
832 blocks x 256 threads, 1 GiB input. Native O3 HIP/C++20, HIP logical
GPU 0, PCI 0000:31:00.0, 104 CUs, CPU+memory NUMA node 0, no model/context,
automatic clocks. The target does not enter the current top-12 finalist
selection, so these target rates are the original three-batch screening
medians (12 reads/batch), not nine-round finalist confirmation medians.

| Run | Target median GB/s | Batch range GB/s |
|---|---:|---:|
| Fresh rebuild, process A | 1406.26 | 1404.66–1416.82 |
| Fresh rebuild, process B | 1405.67 | 1405.27–1410.07 |
| Preserved historical executable | 1412.82 | 1408.05–1420.67 |

Historical final-1 target confirmation median was approximately 1530.45
from nine batches. It is not reproduced in the current machine state.
Fresh sweep best 1 GiB confirmed medians are 1456.94/1465.10; best 4 GiB
confirmed medians 1424.41/1425.78. Those are different configurations,
not target-kernel results.

Active telemetry in the fresh process runs samples core 1700 MHz and
memory 1600 MHz throughout, memory temperatures about 53–71 C. Idle
rocm-smi low-power warnings are not timed-run clock measurements.

The user reports a full power cycle between the historical result and
current tests. Binary equality rules out a compiler/code-generation change,
not a driver/library/firmware-state or physical-allocation difference.
Power cycling should not change the silicon's fundamental bandwidth limit,
but measured throughput can change with initialization, power/thermal state,
software environment and memory placement. No cause has been isolated;
these tests do not prove the historical repeated measurements were merely
random luck. Treat 1530 as a historical best-case, not a reproducible
current baseline.

Full run artifacts:
`temp/vram-ceiling/post-revert-fresh-a/`
`temp/vram-ceiling/post-revert-fresh-b/`
`temp/vram-ceiling/post-revert-preserved/`

The restored benchmark source was not altered to select a new winner or
force a new timing protocol. The tree was clean at that revert checkpoint;
`3c429e7` subsequently reapplied the saved coalescing experiments at the user's
request. The ceiling benchmark itself remains unchanged.

Curated target rows: `results/post-cycle-target.csv`. This replay predates
the later firmware-average clock collection, so there are no pre-cycle
average-clock measurements with which to establish a clock-state change.
