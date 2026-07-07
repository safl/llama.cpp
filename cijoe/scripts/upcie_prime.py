"""Prime the upcie-cuda backend against a fleet of DMA drives.

Empirically, the very first upcie-cuda 4-drive registry_init inside
llama-completion sometimes has one dev return xnvme_dev_get_geo with
lba_nbytes=0 (loader logs 'opened dev has lba_nbytes=0'). Running
xnvmeperf cuda-run against the same URIs first primes something
(CUDA context init, host_heap alloc, device_heap alloc, dma-buf
export path) that avoids the intermittent open failure. Root cause
still TBD.

Runtime is tiny (default 2 s per drive-set) and the drives must
already be bound to uio_pci_generic.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--pci_addrs", type=str, required=True,
                        help="Comma-separated PCIe URIs to prime; must be uio_pci_generic-bound")
    parser.add_argument("--qdepth", type=int, default=32)
    parser.add_argument("--iosize", type=int, default=131072)
    parser.add_argument("--runtime", type=int, default=2)


def main(args, cijoe):
    addrs = [a.strip() for a in args.pci_addrs.split(",") if a.strip()]
    if not addrs:
        log.error("no addrs")
        return 1
    uris = " ".join(addrs)

    # xnvme_dev_open of an upcie-cuda dev is fragile right after a
    # llama-completion (or any cuInit-heavy process) has just exited:
    # cuInit / cuMemCreate for the shared heaps returns ENOMEM until
    # the driver reclaims whatever state the exiting process was
    # holding. Empirically settles in 5-10 s. Retry the whole
    # xnvmeperf priming call with a growing backoff so that flakiness
    # here does not fail the whole workflow. Root cause TBD.
    delays = [3, 6, 10, 15]
    err = 1
    for k, d in enumerate(delays):
        if k > 0:
            log.info(f"prime attempt {k+1}: sleeping {d}s to let CUDA state settle")
        cijoe.run(f"sleep {d}; sync")
        err, _ = cijoe.run(
            f"xnvmeperf cuda-run {uris} "
            f"--iopattern read --qdepth {args.qdepth} "
            f"--iosize {args.iosize} --runtime {args.runtime}"
        )
        if err == 0:
            return 0
        log.warning(f"prime attempt {k+1} failed (err={err}); will retry after backoff")
    log.error(f"xnvmeperf priming failed against {uris} after {len(delays)} attempts")
    return err
