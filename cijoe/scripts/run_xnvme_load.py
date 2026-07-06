"""
xNVMe-async loader measurement (v1: kernel filesystem via io_uring).

Drops the OS page cache before each iteration, then runs the standalone
xnvme-load tool. In this variant xnvme-load opens the file directly
via xNVMe's io_uring file backend and submits pread commands sized to
the device's ``mdts_nbytes`` up to the configured queue depth.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--xnvme_load_bin", type=str, default="/usr/local/bin/xnvme-load")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--async_be", type=str, default="io_uring")
    parser.add_argument("--qd", type=int, default=64)
    parser.add_argument("--io_sizes", type=str, default="",
                        help="comma-separated io sizes in bytes; empty means default")
    parser.add_argument("--iterations", type=int, default=3)


def main(args, cijoe):
    sizes = [int(s) for s in args.io_sizes.split(",")] if args.io_sizes else [0]

    for io_size in sizes:
        io_size_arg = f" --io-size {io_size}" if io_size else ""
        log.info(f"=== xnvme-load io_size={io_size or 'default'} ===")

        for i in range(args.iterations):
            err, _ = cijoe.run("sync && echo 3 > /proc/sys/vm/drop_caches")
            if err:
                log.error(f"iter {i}: drop_caches failed")
                return err

            cmd = (
                f"{args.xnvme_load_bin} "
                f"--file {args.file_path} "
                f"--async {args.async_be} "
                f"--qd {args.qd}"
                f"{io_size_arg}"
            )
            err, _ = cijoe.run(cmd)
            if err:
                log.error(f"io_size={io_size} iter {i}: xnvme-load exited nonzero")
                return err

    return 0
