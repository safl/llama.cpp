"""
Transition a drive from kernel-attached (XFS-mounted) to
uio_pci_generic-bound for upcie-cuda P2P.

Pair with setup_dma_target --layout xfs-fiemap --defer_detach true.
That step stages a drive with XFS + one file and captures FIEMAP,
but leaves the drive mounted so baseline loaders can read the same
file via the kernel path. When the baselines are done, this script
finishes the transition: umount, then DRIVER_OVERRIDE=uio_pci_generic
xnvme-driver.

Idempotent: skips gracefully if the drive is already bound to
uio_pci_generic.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--dma_pci_addr", type=str, required=True,
                        help="PCIe address of the DMA target NVMe, e.g. 0000:c1:00.0")
    parser.add_argument("--fs_mount_point", type=str, required=True,
                        help="Mount point that setup_dma_target left mounted")


def _normalize_pci(addr):
    return addr if addr.count(":") == 2 else f"0000:{addr}"


def _is_bound_to(cijoe, pci_addr, driver):
    err, state = cijoe.run(f"lspci -k -s {pci_addr}")
    if err:
        return False
    return f"Kernel driver in use: {driver}" in state.output()


def main(args, cijoe):
    pci_addr = _normalize_pci(args.dma_pci_addr)

    if _is_bound_to(cijoe, pci_addr, "uio_pci_generic"):
        log.info(f"{pci_addr} already bound to uio_pci_generic; skipping")
        return 0

    err, _ = cijoe.run(f"umount {args.fs_mount_point} 2>/dev/null || true")
    if err:
        return err

    err, _ = cijoe.run(
        f"DRIVER_OVERRIDE=uio_pci_generic PCI_WHITELIST='{pci_addr}' xnvme-driver"
    )
    if err:
        log.error(f"xnvme-driver bind of {pci_addr} to uio_pci_generic failed")
        return err

    err, _ = cijoe.run(f"lspci -k -s {pci_addr}")
    return err
