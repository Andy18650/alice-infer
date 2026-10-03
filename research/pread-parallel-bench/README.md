# Parallel pread bandwidth benchmark

Quick host-IO probe for the data-loader path: how fast can we `pread()` a
large file into anonymous `MADV_HUGEPAGE` staging buffers with N threads?

## Build

```bash
g++ -std=c++20 -O2 -pthread -o pread-parallel-bench bench.cpp
```

## Run

```bash
./pread-parallel-bench <file> <threads>
```

Example:

```bash
./pread-parallel-bench /opt/models/alice/deepseek-r1-distill-llama-8b/weights.bin 4
```

## What it measures

- Each thread gets an evenly-sized slice of the file.
- It `mmap()`s an anonymous buffer, asks for `MADV_HUGEPAGE`, then `pread()`s
  its slice.
- Per-thread bytes, elapsed time, and bandwidth are printed.
- An overall wall-clock time and aggregate bandwidth are printed.

The optimistic case (even division, no coordination) shows the bandwidth
ceiling available if ranks read disjoint records concurrently. If a single
thread is already close to the raw flash speed (~7.4 GB/s on this box), a
single-chunk shared cache that serializes reads would be less painful.

## Sample results

`/opt/models/alice/deepseek-r1-distill-llama-8b/weights.bin` (14.97 GiB),
warm cache:

| threads | per-thread BW | aggregate BW |
|--------:|--------------:|-------------:|
| 1       | ~6.1 GB/s     | ~6.0 GB/s    |
| 2       | ~5.6 GB/s     | ~10.8 GB/s   |
| 4       | ~5.4–7.7 GB/s | ~20.8 GB/s   |
| 8       | ~4.5–6.7 GB/s | ~35.0 GB/s   |

`/opt/models/alice/deepseek-r1-distill-llama-70b/weights.bin` (131.42 GiB),
cold-ish single-thread read: ~6.0 GB/s.

Interpretation: a single `pread` stream is in the 6 GB/s ballpark, slightly
below the ~7.4 GB/s raw rating. With multiple threads the aggregate scales
well because the data is cache-resident, but a single-chunk cache would cap
per-record reads at roughly one stream's speed.
