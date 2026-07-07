"""
Launch N llama.cpp instances concurrently and measure aggregate
wall-clock + GPU PCIe RX.

Works for any loader: pass --loader mmap for the kernel mmap
baseline, --loader stream for --no-mmap, or --loader xnvme with
--xnvme_be io_uring_file (kernel-path xnvme) or upcie-cuda
(userspace P2P). Only the last uses --dma_uris and --dma_extents;
for the kernel baselines they can be empty.

Parallel arrays convention: --model_paths is required; when set,
--dma_uris, --dma_extents, --cpus must either be empty or have
the same length as --model_paths, and index i is instance i's
config.

Wall-clock is the /usr/bin/time-measured duration of the whole
batch (just-before-launch to all-instances-done). nvidia-smi's
pcie.rx.mibps is sampled at ~1 Hz in the background so aggregate
GPU input during the load is visible.
"""

import logging as log
import shlex
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--llama_bin", type=str, required=True)
    parser.add_argument("--n_gpu_layers", type=int, default=999)
    parser.add_argument("--prompt", type=str, default="hi")
    parser.add_argument("--n_tokens", type=int, default=1)
    parser.add_argument("--context_size", type=int, default=4096,
                        help="Per-instance context size (llama.cpp -c). Smaller shrinks the KV cache; useful when packing many instances into one GPU.")
    parser.add_argument("--model_paths", type=str, required=True,
                        help="Comma-separated list of .gguf paths (one per instance)")
    parser.add_argument("--dma_uris", type=str, default="",
                        help="Comma-separated PCIe URIs; empty for non-upcie-cuda loaders")
    parser.add_argument("--dma_extents", type=str, default="",
                        help="Comma-separated extents JSON paths; empty for non-upcie-cuda loaders")
    parser.add_argument("--cpus", type=str, default="",
                        help="Comma-separated CPU indices for LLAMA_XNVME_CPU; empty to skip pinning")
    parser.add_argument("--loader", type=str, default="xnvme",
                        help="mmap | stream | xnvme | default")
    parser.add_argument("--xnvme_be", type=str, default="upcie-cuda",
                        help="Only used with --loader xnvme")
    parser.add_argument("--direct_io", type=str, default="false",
                        help="'true' to pass -dio; only for file-backed xnvme backends")
    parser.add_argument("--log_dir", type=str, default="/tmp/parallel-run",
                        help="Where to write per-instance stdout and the GPU sample trace")


def _split(csv):
    return [x.strip() for x in csv.split(",") if x.strip()]


def _truthy(v):
    return str(v).lower() in ("true", "1", "yes")


def main(args, cijoe):
    models  = _split(args.model_paths)
    uris    = _split(args.dma_uris)
    extents = _split(args.dma_extents)
    cpus    = _split(args.cpus)

    n = len(models)
    for label, xs in [("dma_uris", uris), ("dma_extents", extents), ("cpus", cpus)]:
        if xs and len(xs) != n:
            log.error(f"length mismatch: model_paths={n}, {label}={len(xs)}")
            return 1

    log.info(f"launching {n} instance(s) in parallel; loader={args.loader}")

    err, _ = cijoe.run(f"mkdir -p {args.log_dir} && rm -f {args.log_dir}/*")
    if err:
        log.error(f"could not prepare log_dir={args.log_dir}")
        return err

    loader_flag   = f"--loader {args.loader}" if args.loader != "default" else ""
    xnvme_be_flag = f"--xnvme-be {args.xnvme_be}" if args.loader == "xnvme" and args.xnvme_be else ""
    dio_flag      = "-dio" if _truthy(args.direct_io) else ""

    launch_lines = []
    for i in range(n):
        env_parts = []
        if cpus:
            env_parts.append(f"LLAMA_XNVME_CPU={cpus[i]}")
        if uris:
            env_parts.append(f"LLAMA_XNVME_DMA_URI={uris[i]}")
        if extents:
            env_parts.append(f"LLAMA_XNVME_DMA_EXTENTS={extents[i]}")
        env_prefix = " ".join(env_parts)
        if env_prefix:
            env_prefix += " "

        cmd = (
            f"{env_prefix}"
            f"{args.llama_bin} "
            f"-m {models[i]} "
            f"-ngl {args.n_gpu_layers} "
            f"-c {args.context_size} "
            f"-p {shlex.quote(args.prompt)} "
            f"-n {args.n_tokens} "
            f"-no-cnv "
            f"{loader_flag} {xnvme_be_flag} {dio_flag} "
            f"</dev/null "
            f">{args.log_dir}/inst-{i}.log 2>&1 &"
        )
        launch_lines.append(cmd)

    # `nvidia-smi dmon -s p` prints per-second PCIe rx/tx throughput in
    # MB/s and is available on every recent driver. `--query-gpu=pcie.rx.mibps`
    # is data-center-SKU only.
    sampler = (
        f"( "
        f"  rm -f {args.log_dir}/done; "
        f"  ( nvidia-smi dmon -s p -o T & "
        f"    echo $! > {args.log_dir}/dmon.pid; "
        f"    wait ) > {args.log_dir}/nvidia-smi.log 2>&1 "
        f") & echo $! > {args.log_dir}/sampler.pid"
    )

    prelude = (
        f"sync && echo 3 > /proc/sys/vm/drop_caches && "
        f"{sampler} && "
        f"sleep 1"
    )

    timed_block_body = " ".join(launch_lines) + " wait"
    timed_block = (
        f"/usr/bin/time -f 'PARALLEL_WALLCLOCK_SECONDS: %e' "
        f"bash -c {shlex.quote(timed_block_body)}"
    )

    postlude = (
        f"touch {args.log_dir}/done && "
        f"( dp=$(cat {args.log_dir}/dmon.pid 2>/dev/null); kill $dp 2>/dev/null || true; "
        f"  sp=$(cat {args.log_dir}/sampler.pid); wait $sp 2>/dev/null || true ) && "
        f"echo '--- per-instance stdout tails ---' && "
        f"for f in {args.log_dir}/inst-*.log; do "
        f"  echo \"=== $f ===\"; tail -30 \"$f\"; "
        f"done && "
        f"echo '--- pcie throughput (MB/s, dmon -s p) ---' && "
        f"cat {args.log_dir}/nvidia-smi.log"
    )

    full = f"{prelude} && {timed_block}; rc=$?; {postlude}; exit $rc"

    err, _ = cijoe.run(full)
    return err
