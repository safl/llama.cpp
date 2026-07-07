"""Create a symlink directory that presents shards on several mounts as one directory.

llama.cpp discovers sibling shards of a split GGUF by walking the
``basename-NNNNN-of-MMMMM.gguf`` pattern in the same directory. When
each shard lives on its own mount for parallel-drive load, we still
need one directory that presents all N shards side by side. This
script creates that directory of symlinks; the kernel resolves each
symlink to the underlying mount at open time, so reads land on the
right drive.
"""

import logging as log
from argparse import ArgumentParser


def _split(csv):
    return [x.strip() for x in csv.split(",") if x.strip()]


def add_args(parser: ArgumentParser):
    parser.add_argument("--symlink_dir", type=str, required=True,
                        help="Directory to (re)create, populated with N symlinks")
    parser.add_argument("--basename", type=str, required=True,
                        help="Shared basename for the split (basename-NNNNN-of-NNNNN.gguf)")
    parser.add_argument("--shard_paths", type=str, required=True,
                        help="Comma-separated absolute paths to each shard file, in order 1..N")


def main(args, cijoe):
    shards = _split(args.shard_paths)
    n = len(shards)
    if n < 2:
        log.error("--shard_paths must list at least 2 files")
        return 1

    # Recreate the symlink dir from scratch so stale links from a prior
    # workflow don't hide bugs. Bind-mounts and real files still get
    # unlinked cleanly by `rm -rf` on a symlink dir.
    err, _ = cijoe.run(f"rm -rf {args.symlink_dir}")
    if err:
        return err
    err, _ = cijoe.run(f"mkdir -p {args.symlink_dir}")
    if err:
        return err

    for i, shard in enumerate(shards, start=1):
        target = f"{args.symlink_dir}/{args.basename}-{i:05d}-of-{n:05d}.gguf"
        err, _ = cijoe.run(f"ln -sf {shard} {target}")
        if err:
            log.error(f"failed to symlink {shard} -> {target}")
            return err

    err, _ = cijoe.run(f"ls -la {args.symlink_dir}")
    return err
