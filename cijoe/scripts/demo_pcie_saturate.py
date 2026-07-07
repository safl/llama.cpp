"""
Demo: saturate the GPU's PCIe input by streaming from N NVMes in parallel.

Binds a configurable set of NVMe drives to uio_pci_generic, then runs
xnvmeperf cuda-run against progressively larger subsets (1, 2, 4, N)
so the throughput scaling is visible in a single output. Captures
nvidia-smi's pcie.rx.mibps in parallel to make the aggregate PCIe
utilization concrete.

Best drives to use are the ones sitting under the same root complex as
the GPU (fewest hops for peer-to-peer). On warp: 0000:82:00.0,
0000:83:00.0, 0000:84:00.0, 0000:85:00.0 (Samsung 980 PROs under
0000:80, same as the RTX A6000).
"""

import logging as log
import time
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--pci_addrs", type=str,
                        default="0000:82:00.0,0000:83:00.0,0000:84:00.0,0000:85:00.0",
                        help="Comma-separated PCIe URIs of NVMe drives to enroll in the demo")
    parser.add_argument("--qdepth", type=int, default=64)
    parser.add_argument("--iosize", type=int, default=131072)
    parser.add_argument("--runtime", type=int, default=5)


def _lspci_bound_to(cijoe, addr, driver):
    err, state = cijoe.run(f"lspci -k -s {addr}")
    if err:
        return False
    return f"Kernel driver in use: {driver}" in state.output()


def _bind(cijoe, addrs):
    """Ensure every address is bound to uio_pci_generic."""
    to_bind = [a for a in addrs if not _lspci_bound_to(cijoe, a, "uio_pci_generic")]
    if not to_bind:
        log.info("all drives already bound to uio_pci_generic")
        return 0

    for a in to_bind:
        cijoe.run(f"umount /dev/nvme*n* 2>/dev/null || true")

    whitelist = " ".join(to_bind)
    err, _ = cijoe.run(
        f"DRIVER_OVERRIDE=uio_pci_generic PCI_WHITELIST='{whitelist}' xnvme-driver"
    )
    if err:
        log.error(f"binding {whitelist} to uio_pci_generic failed")
        return err
    return 0


def _run_stage(cijoe, addrs, qdepth, iosize, runtime, label):
    log.info(f"=== {label}: {len(addrs)} drive(s) -> {', '.join(addrs)} ===")
    uris = " ".join(addrs)
    # Run xnvmeperf cuda-run and also sample nvidia-smi's PCIe RX counter
    # a couple of times during the run so the aggregate is visible.
    err, _ = cijoe.run(
        f"( xnvmeperf cuda-run {uris} "
        f"    --iopattern read --qdepth {qdepth} --iosize {iosize} --runtime {runtime} ) & "
        f"XPID=$!; "
        f"sleep 1; "
        f"nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.rx.mibps "
        f"           --format=csv 2>/dev/null; "
        f"sleep 2; "
        f"nvidia-smi --query-gpu=pcie.rx.mibps --format=csv 2>/dev/null; "
        f"wait $XPID"
    )
    return err


def main(args, cijoe):
    addrs = [a.strip() for a in args.pci_addrs.split(",") if a.strip()]
    if not addrs:
        log.error("no PCIe addresses provided")
        return 1

    if _bind(cijoe, addrs):
        return 1

    # Progressive stages: 1, 2, 4 (or N if less than 4), N drives.
    stages = []
    if len(addrs) >= 1: stages.append(addrs[:1])
    if len(addrs) >= 2: stages.append(addrs[:2])
    if len(addrs) >= 4: stages.append(addrs[:4])
    if len(addrs) > 4:  stages.append(addrs)

    for i, subset in enumerate(stages):
        label = f"stage {i+1}"
        err = _run_stage(cijoe, subset, args.qdepth, args.iosize,
                         args.runtime, label)
        if err:
            log.error(f"stage {i+1} failed")
            return err

    return 0
