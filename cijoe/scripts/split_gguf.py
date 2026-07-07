"""Split a monolithic .gguf into N shards via llama-gguf-split.

Reproduces llama.cpp's split-shard naming convention
(``basename-00001-of-00004.gguf`` etc.) on a mounted output directory.
Skips the (slow) split if all expected shards are already present at
their target sizes; this makes iterating on downstream steps cheap.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--llama_bin", type=str,
                        default="/root/llama.cpp/build/bin/llama-gguf-split")
    parser.add_argument("--source_gguf", type=str, required=True,
                        help="Path to the monolithic .gguf to split")
    parser.add_argument("--dst_prefix", type=str, required=True,
                        help="Output basename (without the '-NNNNN-of-MMMMM.gguf' suffix)")
    parser.add_argument("--n_shards", type=int, required=True,
                        help="Number of shards to produce (llama-gguf-split's --split-max-tensors is derived from this)")


def _expected_shard(dst_prefix, i, n):
    return f"{dst_prefix}-{i:05d}-of-{n:05d}.gguf"


def _sum_shard_sizes(cijoe, dst_prefix, n):
    total = 0
    for i in range(1, n + 1):
        err, state = cijoe.run(f"stat -c %s {_expected_shard(dst_prefix, i, n)} 2>/dev/null || echo MISSING")
        if err:
            return None
        out = state.output().strip().splitlines()[-1]
        if out == "MISSING":
            return None
        total += int(out)
    return total


def main(args, cijoe):
    n = args.n_shards
    if n < 2:
        log.error("--n_shards must be at least 2")
        return 1

    err, state = cijoe.run(f"stat -c %s {args.source_gguf}")
    if err:
        log.error(f"source .gguf missing: {args.source_gguf}")
        return err
    src_size = int(state.output().strip().splitlines()[-1])

    # Idempotency: if the shards already exist and their sizes sum to the
    # source size, skip.
    have = _sum_shard_sizes(cijoe, args.dst_prefix, n)
    if have is not None and have == src_size:
        log.info(f"{n} shard(s) already present at {args.dst_prefix}-*-of-*.gguf, skipping split")
        return 0

    # Derive per-shard size budget. llama-gguf-split's greedy packer starts
    # a new shard whenever the *next* tensor would exceed --split-max-size,
    # so if we set the budget to exactly ceil(src_size / n) GiB the packer
    # can spill into an (n+1)th shard on tight fits. Adding a 1 GiB slack
    # keeps ceil(src_size / (budget)) == n for typical llama.cpp weight
    # layouts without producing suspiciously oversized shards.
    per_shard_bytes = (src_size + n - 1) // n
    per_shard_gib   = ((per_shard_bytes + (1 << 30) - 1) // (1 << 30)) + 1

    log.info(f"splitting {args.source_gguf} ({src_size} bytes) into {n} shards "
             f"of ~{per_shard_gib} GiB each -> {args.dst_prefix}-NNNNN-of-{n:05d}.gguf")
    err, _ = cijoe.run(
        f"{args.llama_bin} --split --split-max-size {per_shard_gib}G "
        f"{args.source_gguf} {args.dst_prefix}"
    )
    if err:
        log.error("llama-gguf-split failed")
        return err

    # Verify the produced shard count matches expectation.
    have = _sum_shard_sizes(cijoe, args.dst_prefix, n)
    if have is None:
        log.error(f"expected {n} shard(s) not produced under {args.dst_prefix}-*-of-{n:05d}.gguf")
        return 1
    log.info(f"produced {n} shards totaling {have} bytes")
    return 0
