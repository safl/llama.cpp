# Loader experiments 2026-08-08

Cold-start comparison of `llama-completion`'s four loader paths against a
42 GB Llama-3.3-70B-Instruct Q4_K_M model on the `warp` host: mmap, stream,
xnvme (`io_uring_file` backend, `O_DIRECT`) and upcie-cuda (P2P from NVMe
BAR straight into the tensor's VRAM). Two experiments:

- `reports/exp1_nonsharded/` -- unsharded 42 GB model on one 990 PRO
- `reports/exp2_sharded/` -- 4-shard split across four 990 PROs (one shard per drive)

Same binary drives every measurement, MPS off in both experiments to keep
loader identity unambiguous. Run via the cijoe workflows
`cijoe/tasks/exp1_nonsharded.yaml` and `cijoe/tasks/exp2_sharded.yaml`.

## Headline numbers

The figures quoted externally, all from `exp2` (4 drives, sharded), comparing
`upcie-cuda` against the two kernel loaders. Both columns are given because the
two answer different questions, and quoting one without saying which is
misleading:

| comparison         | load-time speedup | wallclock speedup |
|--------------------|-------------------|-------------------|
| upcie-cuda vs mmap | 5.79x             | 2.97x             |
| upcie-cuda vs stream | 3.99x           | 2.19x             |

Absolute times: upcie-cuda loads the 42 GB model in 2,631 ms (6.13 s wallclock),
against mmap at 15,234 ms (18.20 s) and stream at 10,504 ms (13.43 s).

`load_time` is llama.cpp's own timing of the weight upload. `wallclock` covers
the whole `llama-completion` invocation and therefore also pays roughly 3.3 s of
fixed per-run overhead that is near-constant across loaders, which dilutes the
ratio. Which column is the right one depends on the caller: a long-running
server loads once and then serves many prompts, so it reads load time; a
one-shot CLI reads wallclock. See "Load time vs total wallclock" below.

The speedups against `stream` are derived from the same `exp2` table; the tables
themselves are normalised against `mmap`.

## Results

Numbers are `load time` (from llama.cpp's `common_perf_print`, the weight
upload only) and `wallclock` (`/usr/bin/time -f %e` around the entire
`llama-completion` invocation, including CUDA context init, prefill of a
two-token prompt, one decode step, and shutdown). Every speedup below is
labelled as either "load-time speedup" or "wallclock speedup" so the two
are never confused; see the "Load time vs total wallclock" section for
what the constant non-loader overhead looks like on this host.

### exp1: 1 drive, unsharded

| loader             | load\_time  | wallclock  | RSS      | load-time speedup vs mmap | wallclock speedup vs mmap |
|--------------------|-------------|------------|----------|---------------------------|---------------------------|
| mmap               | 15,179 ms   | 18.15 s    | 42.0 GB  | 1.00x                     | 1.00x                     |
| stream             | 12,418 ms   | 15.37 s    | 1.07 GB  | 1.22x                     | 1.18x                     |
| xnvme io\_uring    | 8,731 ms    | 11.67 s    | 1.54 GB  | 1.74x                     | 1.55x                     |
| upcie-cuda         | 6,822 ms    | 10.24 s    | 1.17 GB  | 2.22x                     | 1.77x                     |

### exp2: 4 drives, sharded

| loader             | load\_time  | wallclock  | RSS      | load-time speedup vs mmap | wallclock speedup vs mmap |
|--------------------|-------------|------------|----------|---------------------------|---------------------------|
| mmap               | 15,234 ms   | 18.20 s    | 42.0 GB  | 1.00x                     | 1.00x                     |
| stream             | 10,504 ms   | 13.43 s    | 1.07 GB  | 1.45x                     | 1.36x                     |
| xnvme io\_uring    | 8,719 ms    | 11.64 s    | 1.54 GB  | 1.75x                     | 1.56x                     |
| upcie-cuda         | 2,631 ms    | 6.13 s     | 1.17 GB  | 5.79x                     | 2.97x                     |

### 1 drive vs 4 drives, per loader

Same source, same loader, same content, only difference is the number of
NVMe drives the reads go through. Load-time scaling isolates the loader's
own drive-count parallelism; wallclock scaling shows what a caller of the
whole binary observes.

| loader             | 1 drive load / wall | 4 drives load / wall | load-time speedup (4/1) | wallclock speedup (4/1) |
|--------------------|---------------------|----------------------|-------------------------|-------------------------|
| mmap               | 15,179 ms / 18.15 s | 15,234 ms / 18.20 s  | 1.00x                   | 1.00x                   |
| stream             | 12,418 ms / 15.37 s | 10,504 ms / 13.43 s  | 1.18x                   | 1.14x                   |
| xnvme io\_uring    | 8,731 ms / 11.67 s  | 8,719 ms / 11.64 s   | 1.00x                   | 1.00x                   |
| upcie-cuda         | 6,822 ms / 10.24 s  | 2,631 ms / 6.13 s    | 2.59x                   | 1.67x                   |

Only upcie-cuda actually parallelises reads across the four drives. mmap
and xnvme `io_uring_file` are flat: mmap because the VFS + page-cache layer
serialises regardless of physical drive count, and xnvme `io_uring_file`
because the loader iterates shards serially at the file level (one file at
a time, only one drive active at any moment). Stream gets a small boost
from readahead across separate mount points but is still bound by
sequential file traversal. Upcie-cuda's per-file drive routing keeps all
four drives busy at once and cuts load\_time by 4,191 ms compared to its
own single-drive baseline; the per-run `submit+drain` summary in each
upcie `cmd_02.output` reports the aggregate GiB/s directly (~6.5 at one
drive, ~21 at four).

### Load time vs total wallclock

The `load_time` column is `llama.cpp`'s internal timing of the weight
upload; `wallclock` is `/usr/bin/time -f '%e'` around the entire
`llama-completion` invocation and therefore also includes process spawn,
`libcuda.so` dlopen, CUDA context init, GGUF header + tensor table read,
KV cache allocation, prefill of the two-token prompt, one decode step,
and cleanup. On this host that non-loader overhead is roughly 3.3 s per
run and is roughly constant across loaders and drive counts.

That fixed overhead dilutes the loader speedup when measured on wallclock
instead of on load\_time. Upcie-cuda's 5.79x load-time speedup at four
drives shows the loader is 5.79x faster at the job it is measuring; the
same run's 2.97x wallclock speedup shows what a caller of the whole
binary observes.

If the eventual use is a long-running server that loads the model once
and then serves many prompts, the load-time speedup is the right column
to read. If the use is a one-shot CLI per prompt where the process comes
up, loads, generates, and exits, the wallclock speedup is the right one.
Every table above lists both, always labelled explicitly, so a reader
never has to guess.

### Why the kernel loaders do not scale

mmap and stream are built on blocking, synchronous system calls: mmap
page-faults each tensor's bytes in on first touch, stream `read()`s each
tensor sequentially. To keep more than one drive busy at a time they would
need a worker thread per file plus shared state around pinned host buffers,
the CUDA upload stream, and progress counters. That is the mechanism, and
it costs real work: mutexes on the tensor-set path, an N-way ring of
pinned buffers or contention on a single one, careful sync of async H->D
uploads issued from multiple producer threads, and atomics on `size_done`
for the progress callback. Every drive added is another thread and another
locked resource.

The xnvme path is async from the start. Each submit posts a request to a
ring; the completion callback fires later and kicks the next CUDA copy.
Fanning across N drives is N queues, not N threads, and each new drive
adds queue slots rather than mutex traffic. That is why the upcie-cuda
number scales cleanly from 6.8 to 2.6 s and why extending the file-backed
xnvme variant to do the same routing would be mechanical, whereas adding
per-drive parallelism to mmap or stream is a genuine reengineering.

## Reproducing

On the target (`warp`), with the four DMA drives on `0000:c1..c4:00.0`
bound to `nvme`, the source `.gguf` staged at
`/mnt/gguf-src/Llama-3.3-70B-Instruct-Q4_K_M.gguf`, and a
sharded copy under
`/mnt/gguf-src/Llama-3.3-70B-Q4_K_M-*-of-00004.gguf`:

```
cijoe -c cijoe/configs/warp.toml cijoe/tasks/exp1_nonsharded.yaml -o cijoe/reports/exp1_nonsharded
cijoe -c cijoe/configs/warp.toml cijoe/tasks/exp2_sharded.yaml    -o cijoe/reports/exp2_sharded
```

Each workflow starts with `disable_mps`, stages XFS on the target drives,
runs the kernel loaders (mmap, stream, xnvme `io_uring_file`), detaches the
drives to `uio_pci_generic`, primes upcie-cuda with `xnvmeperf cuda-run` to
work around an intermittent `xnvme_dev_open ENOMEM` right after a
cuInit-heavy process exits, and finally runs upcie-cuda. `run_baseline`
fail-fasts on a silent stream fallback so a loader that quietly downgrades
does not misreport a stream number under its labelled loader.
