"""
Baseline cold-start measurement for llama.cpp.

Drops the OS page cache before each iteration, runs llama-completion
with ``-n 1`` so it stops as soon as one token is produced, and
records the wall-clock time from process start to exit. That interval
is dominated by mmap + host->device weight copies, so it is a proxy
for "seconds to first token".

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
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--no_mmap", type=str, default="false",
                        help="'true' to pass --no-mmap to llama-completion")


def main(args, cijoe):
    no_mmap_flag = ""
    if str(args.no_mmap).lower() in ("true", "1", "yes"):
        no_mmap_flag = " --no-mmap"

    for i in range(args.iterations):
        err, _ = cijoe.run("sync && echo 3 > /proc/sys/vm/drop_caches")
        if err:
            log.error(f"iter {i}: drop_caches failed")
            return err

        cmd = (
            f"/usr/bin/time -f 'wallclock: %e s | rss %M KiB' "
            f"{args.llama_bin} "
            f"-m {args.gguf_path} "
            f"-ngl {args.n_gpu_layers} "
            f"-p {shlex.quote(args.prompt)} "
            f"-n {args.n_tokens} "
            f"-no-cnv"
            f"{no_mmap_flag} "
            f"< /dev/null"
        )
        err, _ = cijoe.run(cmd)
        if err:
            log.error(f"iter {i}: llama-cli exited nonzero")
            return err

    return 0
