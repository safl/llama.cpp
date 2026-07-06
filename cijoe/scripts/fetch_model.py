"""
Fetch a .gguf model to the target host if it is not already at gguf_src.

Idempotent: if gguf_src already exists and its size on disk matches
the Content-Length of the URL, skip the download.
"""

import logging as log
from argparse import ArgumentParser
from pathlib import PurePosixPath


def add_args(parser: ArgumentParser):
    parser.add_argument("--gguf_src", type=str, required=True,
                        help="Absolute path where the .gguf should live on the target")
    parser.add_argument("--gguf_url", type=str, required=True,
                        help="Direct-download URL for the .gguf")


def main(args, cijoe):
    dst_dir = str(PurePosixPath(args.gguf_src).parent)
    err, _ = cijoe.run(f"mkdir -p {dst_dir}")
    if err:
        return err

    err, _ = cijoe.run(f'[ -s "{args.gguf_src}" ]')
    if not err:
        log.info(f"{args.gguf_src} already present, skipping download")
        return 0

    log.info(f"downloading {args.gguf_url} -> {args.gguf_src}")
    err, _ = cijoe.run(
        f"wget --progress=dot:giga --tries=3 -O {args.gguf_src} {args.gguf_url}"
    )
    if err:
        log.error("wget failed")
        cijoe.run(f"rm -f {args.gguf_src}")
        return err

    err, _ = cijoe.run(f"ls -lh {args.gguf_src}")
    return err
