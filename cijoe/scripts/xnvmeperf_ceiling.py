"""Run xnvmeperf cuda-run against a fleet of DMA drives to characterise the aggregate P2P ceiling.

Drives must already be bound to uio_pci_generic. Progressively runs
1, 2, 4 drives so the scaling is visible. Runtime is short (a few
seconds per stage). Report the aggregate MiB/s from the tail line
of each xnvmeperf stage output.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--pci_addrs", type=str, required=True,
                        help="Comma-separated PCIe URIs of drives to hit (must already be uio_pci_generic-bound)")
    parser.add_argument("--qdepth", type=int, default=64)
    parser.add_argument("--iosize", type=int, default=131072)
    parser.add_argument("--runtime", type=int, default=5)


def _split(csv):
    return [x.strip() for x in csv.split(",") if x.strip()]


def main(args, cijoe):
    addrs = _split(args.pci_addrs)
    if not addrs:
        log.error("no addrs")
        return 1

    stages = []
    stages.append(addrs[:1])
    if len(addrs) >= 2:
        stages.append(addrs[:2])
    if len(addrs) >= 4:
        stages.append(addrs[:4])
    if len(addrs) > 4:
        stages.append(addrs)

    for i, subset in enumerate(stages):
        cijoe.run(f"echo '=== stage {i+1}: {len(subset)} drive(s) ==='")
        uris = " ".join(subset)
        err, _ = cijoe.run(
            f"xnvmeperf cuda-run {uris} "
            f"--iopattern read --qdepth {args.qdepth} "
            f"--iosize {args.iosize} --runtime {args.runtime}"
        )
        if err:
            log.error(f"stage {i+1} failed")
            return err

    return 0
