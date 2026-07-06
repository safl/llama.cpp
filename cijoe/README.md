# cijoe: cold-start LLM benchmarks

This subtree is a self-contained cijoe workflow that measures the
wall-clock cost of getting a `.gguf` model's weights from an NVMe disk
into GPU device memory, for two data paths against the same physical
bytes:

1. **Baseline.** `llama.cpp`'s default loader, which mmaps the file and
   copies each tensor into device memory via `cudaMemcpyAsync`.
2. **xNVMe (v1).** A standalone `xnvme-load` tool (source under
   `tools/xnvme-load/`) that opens the file through xNVMe's io_uring
   file backend and submits async preads sized to the device's
   `mdts_nbytes` up to a configurable queue depth, into a DMA-capable
   host buffer.

Both paths read the same physical bytes from the same file on the same
filesystem. A planned v2 of `xnvme-load` will swap the file backend
for [xal](https://github.com/xnvme/xal) on the raw block device to
drop the kernel out of the data path entirely; the surrounding
workflow and measurements stay the same.

## Layout

    cijoe/
      configs/
        warp.toml            ssh config for the "warp" bench host
        wave.toml            ssh config for the "wave" bench host
      scripts/
        git_sync.py          push local source to the remote host
        remote_build.py      build llama.cpp (CUDA) and xnvme-load on remote
        setup_dataset.py     format XFS on the target NVMe, stage the .gguf
        run_baseline.py      cold-caches + llama-cli, capture wall clock
        run_xnvme_load.py    cold-caches + xnvme-load, capture throughput
      tasks/
        bench_cold_start.yaml  the workflow that chains the above
      tools/
        xnvme-load/
          xnvme_load.c       ~180 LoC xNVMe-async file loader (io_uring)
          meson.build

## Prerequisites on the target host

CUDA toolchain and drivers, `xnvme >= 0.7.0` installed (discoverable
via `pkg-config`), meson, cmake, gcc, xfsprogs, and a spare NVMe
device or partition dedicated to the model. Also, the
`.gguf` file must already be staged on the target at the path given in
`gguf_src`.

## Running

    cd /home/odus/git/llama.cpp/cijoe
    cijoe -c configs/wave.toml tasks/bench_cold_start.yaml

Adjust the `with:` values in the task to match the target's device
names, mount points, and model file location.
