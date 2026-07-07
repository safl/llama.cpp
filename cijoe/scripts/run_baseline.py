"""
Cold-start measurement for llama.cpp with a configurable loader path.

Drops the OS page cache before each iteration, runs llama-completion
with ``-n 1`` so it stops as soon as one token is produced, and
records the wall-clock and max RSS via /usr/bin/time. The interval
is dominated by the weight-load phase, so it is the "seconds to
first token" figure of merit.

The same script drives every loader variant so their cijoe outputs
land side by side under the same step template. Pass:

    loader:    mmap | stream | xnvme | default
    direct_io: true | false
    xnvme_be:  io_uring_file | libaio_file | thrpool_file | ...

Uses llama-completion rather than llama-cli because upstream
refactored llama-cli into a chat-only REPL; llama-completion is the
tty-free one-shot binary.
"""

import logging as log
import shlex
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--llama_bin", type=str, required=True)
    parser.add_argument("--gguf_path", type=str, required=True)
    parser.add_argument("--n_gpu_layers", type=int, default=999)
    parser.add_argument("--prompt", type=str, default="hi")
    parser.add_argument("--n_tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--loader", type=str, default="default",
                        help="mmap | stream | xnvme | default (derives from --mmap flags)")
    parser.add_argument("--direct_io", type=str, default="false",
                        help="'true' to pass -dio to llama-completion")
    parser.add_argument("--xnvme_be", type=str, default="",
                        help="xNVMe backend name, e.g. io_uring_file")
    parser.add_argument("--xnvme_cpu", type=str, default="",
                        help="CPU index to pin the xNVMe loader thread to (via LLAMA_XNVME_CPU)")


def _truthy(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def main(args, cijoe):
    loader_flag = f" --loader {args.loader}" if args.loader != "default" else ""
    dio_flag = " -dio" if _truthy(args.direct_io) else ""
    xnvme_be_flag = f" --xnvme-be {args.xnvme_be}" if args.xnvme_be else ""

    for i in range(args.iterations):
        err, _ = cijoe.run("sync && echo 3 > /proc/sys/vm/drop_caches")
        if err:
            log.error(f"iter {i}: drop_caches failed")
            return err

        env_prefix = f"LLAMA_XNVME_CPU={args.xnvme_cpu} " if args.xnvme_cpu else ""

        cmd = (
            f"{env_prefix}"
            f"/usr/bin/time -f 'wallclock: %e s | rss %M KiB' "
            f"{args.llama_bin} "
            f"-m {args.gguf_path} "
            f"-ngl {args.n_gpu_layers} "
            f"-p {shlex.quote(args.prompt)} "
            f"-n {args.n_tokens} "
            f"-no-cnv"
            f"{loader_flag}{dio_flag}{xnvme_be_flag} "
            f"< /dev/null"
        )
        err, _ = cijoe.run(cmd)
        if err:
            log.error(f"iter {i}: llama-completion exited nonzero")
            return err

    return 0
