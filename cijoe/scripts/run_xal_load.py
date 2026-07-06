"""
xal-based loader measurement.

Drops the OS page cache before each iteration, then runs the standalone
xal-load tool. xal-load opens the NVMe block device via xNVMe, asks
xal for the .gguf's XFS extents on that device, and reads their LBA
ranges through an xNVMe async queue into a DMA-capable host buffer.
This bypasses the kernel filesystem read path entirely; the FS just
sits mounted as a bystander.

Requires the device to be identified by its PCIe address (same as
setup_dataset) so the /dev/nvmeXn1 name is resolved at run time.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--xal_load_bin", type=str,
                        default="/usr/local/bin/xal-load")
    parser.add_argument("--pci_addr", type=str, required=True)
    parser.add_argument("--namespace", type=int, default=1)
    parser.add_argument("--file_path", type=str, required=True,
                        help="path within the mounted FS (used by xal only)")
    parser.add_argument("--qd", type=int, default=64)
    parser.add_argument("--io_sizes", type=str, default="",
                        help="comma-separated io sizes in bytes; empty means default")
    parser.add_argument("--iterations", type=int, default=3)


def _normalize_pci(addr):
    return addr if addr.count(":") == 2 else f"0000:{addr}"


def _resolve_ng_device(cijoe, pci_addr, namespace):
    """Resolve PCIe address to /dev/ngXnY (NVMe namespace char device)
    so xnvme can use the io_uring_cmd passthrough path."""
    pci_addr = _normalize_pci(pci_addr)
    err, state = cijoe.run(f"ls /sys/bus/pci/devices/{pci_addr}/nvme/")
    if err:
        log.error(f"no nvme controller under /sys/bus/pci/devices/{pci_addr}")
        return None
    ctrl = state.output().strip().split()[0]
    # 'nvme3' controller -> 'ng3' char dev family
    ctrl_index = ctrl.replace("nvme", "")
    return f"/dev/ng{ctrl_index}n{namespace}"


def main(args, cijoe):
    dev_uri = _resolve_ng_device(cijoe, args.pci_addr, args.namespace)
    if not dev_uri:
        return 1
    log.info(f"resolved {args.pci_addr} -> {dev_uri}")

    sizes = [int(s) for s in args.io_sizes.split(",")] if args.io_sizes else [0]

    for io_size in sizes:
        io_size_arg = f" --io-size {io_size}" if io_size else ""
        log.info(f"=== xal-load io_size={io_size or 'default'} ===")

        for i in range(args.iterations):
            err, _ = cijoe.run("sync && echo 3 > /proc/sys/vm/drop_caches")
            if err:
                log.error(f"iter {i}: drop_caches failed")
                return err

            cmd = (
                f"{args.xal_load_bin} "
                f"--dev-uri {dev_uri} "
                f"--file {args.file_path} "
                f"--qd {args.qd}"
                f"{io_size_arg}"
            )
            err, _ = cijoe.run(cmd)
            if err:
                log.error(f"io_size={io_size} iter {i}: xal-load exited nonzero")
                return err

    return 0
