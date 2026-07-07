"""
Format a spare NVMe device XFS, mount it, and stage a .gguf model on it.

The XFS partition serves two purposes at once:

- llama.cpp reads the .gguf as a normal mounted-FS file (baseline path).
- A future v2 of xnvme-load parses the XFS on-disk format via xal on the
  raw block device and reads the same file via LBA extents.

Using the same physical bytes for both keeps the comparison honest.

The target SSD is identified by its PCIe address so the workflow stays
stable across reboots that reshuffle /dev/nvmeXn1 names.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--pci_addr", type=str, required=True,
                        help="PCIe address of the target NVMe, e.g. 0000:01:00.0")
    parser.add_argument("--namespace", type=int, default=1,
                        help="NVMe namespace id (1 for typical single-ns SSDs)")
    parser.add_argument("--mount_point", type=str, required=True)
    parser.add_argument("--gguf_src", type=str, default="",
                        help="Model file to copy onto the mount; empty for mount-only")
    parser.add_argument("--gguf_dst_name", type=str, default="model.gguf")
    parser.add_argument("--force_format", type=str, default="false",
                        help="'true' to run mkfs.xfs -f on the resolved device")


def _normalize_pci(addr: str) -> str:
    return addr if addr.count(":") == 2 else f"0000:{addr}"


def _resolve_block_device(cijoe, pci_addr: str, namespace: int):
    pci_addr = _normalize_pci(pci_addr)

    err, state = cijoe.run(f"ls /sys/bus/pci/devices/{pci_addr}/nvme/")
    if err:
        log.error(f"no nvme controller under /sys/bus/pci/devices/{pci_addr}")
        return None

    ctrl = state.output().strip().split()[0]
    return f"/dev/{ctrl}n{namespace}"


def main(args, cijoe):
    target_device = _resolve_block_device(cijoe, args.pci_addr, args.namespace)
    if not target_device:
        return 1
    log.info(f"resolved {args.pci_addr} -> {target_device}")

    cijoe.run(f"umount {args.mount_point} || true")
    cijoe.run(f"umount {target_device} || true")

    if str(args.force_format).lower() in ("true", "1", "yes"):
        err, _ = cijoe.run(f"mkfs.xfs -f {target_device}")
        if err:
            log.error(f"mkfs.xfs on {target_device} failed")
            return err

    err, _ = cijoe.run(f"mkdir -p {args.mount_point}")
    if err:
        return err

    err, _ = cijoe.run(f"mount {target_device} {args.mount_point}")
    if err:
        log.error(f"mount {target_device} at {args.mount_point} failed")
        return err

    if not args.gguf_src:
        log.info(f"mount-only mode (empty gguf_src); {target_device} mounted at {args.mount_point}")
        err, _ = cijoe.run(f"ls -la {args.mount_point}")
        return err

    dst = f"{args.mount_point}/{args.gguf_dst_name}"
    err, _ = cijoe.run(f"cp -v {args.gguf_src} {dst}")
    if err:
        log.error(f"cp {args.gguf_src} -> {dst} failed")
        return err

    err, _ = cijoe.run(f"sync && ls -lh {dst}")
    return err
