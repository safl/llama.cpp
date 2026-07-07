"""
Prepare a spare NVMe as the DMA target for xNVMe's upcie-cuda backend.

Two layout modes are supported. Both end with the drive detached
from the kernel NVMe driver and bound to uio_pci_generic, and both
emit an extents JSON that the loader consumes via
LLAMA_XNVME_DMA_EXTENTS.

  layout=raw (default, backwards compatible):
    Copy the .gguf onto the raw device starting at LBA 0 via dd.
    The drive holds a raw byte image of the file; file offset O
    trivially maps to LBA (O / lba_nbytes). The extents JSON has
    a single entry, [{file_offset: 0, lba: 0, length: file_size}].

  layout=xfs-fiemap:
    Format XFS on the drive, mount, copy the .gguf onto the FS as
    a normal file, run xfs_bmap (FIEMAP equivalent) to capture the
    file's on-disk extents, umount, then detach + bind as above.
    The extents JSON carries one entry per XFS extent with the
    per-drive LBA computed from the physical block offset. This is
    the realistic "user put their model on a real FS" shape.

In both modes the workflow's earlier setup_dataset step must have
staged the model onto a *different* drive (the source drive,
mounted at /mnt/gguf/model.gguf) so llama.cpp can open the file for
GGUF metadata reads during the run; the DMA target here is the only
drive touched during bulk weight reads.
"""

import json
import logging as log
import re
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--dma_pci_addr", type=str, required=True,
                        help="PCIe address of the DMA target NVMe, e.g. 0000:01:00.0")
    parser.add_argument("--dma_namespace", type=int, default=1)
    parser.add_argument("--source_gguf", type=str, required=True,
                        help="Path to the .gguf staged by setup_dataset on the mounted source drive")
    parser.add_argument("--gguf_extents_path", type=str, default="/root/model.extents.json",
                        help="Where to write the extents JSON that the loader consumes")
    parser.add_argument("--layout", type=str, default="raw",
                        choices=["raw", "xfs-fiemap"],
                        help="Placement mode; raw = dd at LBA 0, xfs-fiemap = XFS + FIEMAP extents")
    parser.add_argument("--fs_mount_point", type=str, default="/mnt/dma-target",
                        help="Where to mount the drive during xfs-fiemap setup (transient)")
    parser.add_argument("--fs_dst_name", type=str, default="model.gguf",
                        help="Name of the copied file on the mounted FS (xfs-fiemap only)")
    parser.add_argument("--defer_detach", type=str, default="false",
                        help="'true' to leave the drive mounted after FIEMAP capture; "
                             "detach_dma_target must run before upcie-cuda open (xfs-fiemap only)")


def _normalize_pci(addr):
    return addr if addr.count(":") == 2 else f"0000:{addr}"


def _resolve_block_device(cijoe, pci_addr, namespace):
    pci_addr = _normalize_pci(pci_addr)
    err, state = cijoe.run(f"ls /sys/bus/pci/devices/{pci_addr}/nvme/")
    if err:
        log.error(f"no nvme controller under /sys/bus/pci/devices/{pci_addr}")
        return None
    ctrl = state.output().strip().split()[0]
    return f"/dev/{ctrl}n{namespace}"


def _is_bound_to(cijoe, pci_addr, driver):
    err, state = cijoe.run(f"lspci -k -s {pci_addr}")
    if err:
        return False
    return f"Kernel driver in use: {driver}" in state.output()


def _read_lba_nbytes(cijoe, dev):
    """Read /sys/block/<name>/queue/logical_block_size for the given block device."""
    name = dev.split("/")[-1]  # e.g. nvme0n1
    err, state = cijoe.run(f"cat /sys/block/{name}/queue/logical_block_size")
    if err:
        return None
    try:
        return int(state.output().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _xfs_bmap_extents(cijoe, mounted_file, lba_nbytes):
    """Run xfs_bmap and return a list of dict {file_offset, lba, length}.

    xfs_bmap prints extents in 512-byte basic-block (BB) units:
      0: [0..99999]: 96..100095
      1: [100000..199999]: 2097152..2197151

    We convert to per-drive units: file_offset is bytes from start
    of file, lba is drive LBA (in units of lba_nbytes) at which the
    extent's data resides on the block device, length is bytes.

    Rejects the file if any extent is a "hole" (unallocated) - the
    loader would silently read zeros.
    """
    err, state = cijoe.run(f"xfs_bmap {mounted_file}")
    if err:
        log.error(f"xfs_bmap {mounted_file} failed")
        return None
    extents = []
    for line in state.output().splitlines():
        if "hole" in line.lower():
            log.error(f"xfs_bmap reported a hole in {mounted_file}: {line}")
            return None
        m = re.match(r"\s*\d+:\s*\[(\d+)\.\.(\d+)\]:\s*(\d+)\.\.(\d+)\s*$", line)
        if not m:
            continue
        lstart_bb = int(m.group(1))
        lend_bb   = int(m.group(2))
        pstart_bb = int(m.group(3))
        # BB is fixed at 512 bytes regardless of drive sector size.
        length_bytes  = (lend_bb - lstart_bb + 1) * 512
        file_offset   = lstart_bb * 512
        physical_byte = pstart_bb * 512
        if physical_byte % lba_nbytes != 0:
            log.error(f"extent physical_offset {physical_byte} not aligned to lba_nbytes {lba_nbytes}")
            return None
        lba = physical_byte // lba_nbytes
        extents.append({"file_offset": file_offset, "lba": lba, "length": length_bytes})
    if not extents:
        log.error(f"xfs_bmap returned no extents for {mounted_file}")
        return None
    return extents


def _write_extents_json(cijoe, path, payload):
    err, _ = cijoe.run(
        f"cat > {path} <<'EOF'\n"
        f"{json.dumps(payload, indent=2)}\n"
        f"EOF"
    )
    if err:
        log.error(f"writing extents json to {path} failed")
    return err


def _detach_and_bind_uio(cijoe, pci_addr):
    err, _ = cijoe.run(
        f"DRIVER_OVERRIDE=uio_pci_generic PCI_WHITELIST='{pci_addr}' xnvme-driver"
    )
    if err:
        log.error(f"xnvme-driver bind of {pci_addr} to uio_pci_generic failed")
    return err


def _setup_xfs_fiemap(args, cijoe, pci_addr, dev):
    """XFS + FIEMAP mode: format, mount, cp, xfs_bmap, unmount, detach."""
    lba_nbytes = _read_lba_nbytes(cijoe, dev)
    if not lba_nbytes:
        log.error(f"failed to read logical_block_size for {dev}")
        return 1

    # Belt-and-braces: nothing on this drive should be in use.
    cijoe.run(f"umount {args.fs_mount_point} 2>/dev/null || true")
    cijoe.run(f"umount {dev} 2>/dev/null || true")

    err, _ = cijoe.run(f"mkfs.xfs -f {dev}")
    if err:
        log.error(f"mkfs.xfs on {dev} failed")
        return err

    err, _ = cijoe.run(f"mkdir -p {args.fs_mount_point}")
    if err:
        return err

    err, _ = cijoe.run(f"mount {dev} {args.fs_mount_point}")
    if err:
        log.error(f"mount {dev} at {args.fs_mount_point} failed")
        return err

    mounted_file = f"{args.fs_mount_point}/{args.fs_dst_name}"
    err, _ = cijoe.run(f"cp -v {args.source_gguf} {mounted_file}")
    if err:
        log.error(f"cp {args.source_gguf} -> {mounted_file} failed")
        cijoe.run(f"umount {args.fs_mount_point} || true")
        return err
    cijoe.run(f"sync")

    err, state = cijoe.run(f"stat -c %s {mounted_file}")
    if err:
        log.error("stat on mounted file failed")
        cijoe.run(f"umount {args.fs_mount_point} || true")
        return err
    file_size = int(state.output().strip().splitlines()[-1])

    extents = _xfs_bmap_extents(cijoe, mounted_file, lba_nbytes)
    if extents is None:
        cijoe.run(f"umount {args.fs_mount_point} || true")
        return 1

    payload = {
        "layout": "xfs-fiemap",
        "device": dev,
        "pci_addr": pci_addr,
        "file_size": file_size,
        "lba_nbytes": lba_nbytes,
        "extents": extents,
    }
    err = _write_extents_json(cijoe, args.gguf_extents_path, payload)
    if err:
        cijoe.run(f"umount {args.fs_mount_point} || true")
        return err

    if str(args.defer_detach).lower() in ("true", "1", "yes"):
        log.info(f"{pci_addr}: FIEMAP captured, drive stays mounted at {args.fs_mount_point}; "
                 "run detach_dma_target before opening via upcie-cuda")
        return 0

    err, _ = cijoe.run(f"umount {args.fs_mount_point}")
    if err:
        log.error(f"umount {args.fs_mount_point} failed")
        return err

    err = _detach_and_bind_uio(cijoe, pci_addr)
    if err:
        return err

    err, _ = cijoe.run(f"lspci -k -s {pci_addr}")
    return err


def _setup_raw(args, cijoe, pci_addr):
    """Raw dd mode: image the .gguf onto the drive at LBA 0."""
    # Idempotent shortcut: if the drive is already bound to uio_pci_generic
    # from an earlier workflow run, assume its raw layout still matches the
    # source and skip the (slow) dd + bind. The extents json is still
    # (re)written since it is cheap.
    if _is_bound_to(cijoe, pci_addr, "uio_pci_generic"):
        log.info(f"{pci_addr} is already bound to uio_pci_generic; skipping dd + bind")
        err, state = cijoe.run(f"stat -c %s {args.source_gguf}")
        if err:
            log.error("stat on source .gguf failed")
            return err
        file_size = int(state.output().strip().splitlines()[-1])
        payload = {
            "layout": "raw",
            "device": None,
            "pci_addr": pci_addr,
            "file_size": file_size,
            "lba_nbytes": 512,
            "extents": [
                {"file_offset": 0, "lba": 0, "length": file_size},
            ],
        }
        return _write_extents_json(cijoe, args.gguf_extents_path, payload)

    dev = _resolve_block_device(cijoe, pci_addr, args.dma_namespace)
    if not dev:
        return 1
    log.info(f"DMA target: {pci_addr} -> {dev}")

    # Belt-and-braces: the DMA drive must not have anything mounted on it.
    cijoe.run(f"umount {dev} 2>/dev/null || true")

    err, _ = cijoe.run(
        f"dd if={args.source_gguf} of={dev} bs=4M status=progress conv=fdatasync"
    )
    if err:
        log.error(f"dd {args.source_gguf} -> {dev} failed")
        return err

    err, state = cijoe.run(f"stat -c %s {args.source_gguf}")
    if err:
        log.error("stat on source .gguf failed")
        return err
    file_size = int(state.output().strip().splitlines()[-1])

    lba_nbytes = _read_lba_nbytes(cijoe, dev) or 512
    payload = {
        "layout": "raw",
        "device": dev,
        "pci_addr": pci_addr,
        "file_size": file_size,
        "lba_nbytes": lba_nbytes,
        "extents": [
            {"file_offset": 0, "lba": 0, "length": file_size},
        ],
    }
    err = _write_extents_json(cijoe, args.gguf_extents_path, payload)
    if err:
        return err

    # Detach from kernel + bind to uio_pci_generic. xNVMe's upcie-cuda
    # backend's dev_open explicitly checks the driver name and rejects
    # anything other than uio_pci_generic (see xnvme_be_upcie_cuda_dev.c
    # line 104). DRIVER_OVERRIDE=uio_pci_generic forces xnvme-driver /
    # spdk-driver to use that binding; PCI_WHITELIST scopes the bind to
    # only this device.
    err = _detach_and_bind_uio(cijoe, pci_addr)
    if err:
        return err

    err, _ = cijoe.run(f"lspci -k -s {pci_addr}")
    return err


def main(args, cijoe):
    pci_addr = _normalize_pci(args.dma_pci_addr)

    if args.layout == "xfs-fiemap":
        # Once bound to uio_pci_generic the kernel FS layer is gone so we
        # cannot re-derive FIEMAP. If the extents JSON is already on disk
        # from an earlier run we skip; otherwise reset the drive first
        # (`xnvme-driver reset`) so we can format XFS again.
        if _is_bound_to(cijoe, pci_addr, "uio_pci_generic"):
            err, _ = cijoe.run(f"[ -s {args.gguf_extents_path} ]")
            if not err:
                log.info(f"{pci_addr} already unbound and {args.gguf_extents_path} exists; skipping")
                return 0
            log.error(f"{pci_addr} bound to uio_pci_generic but {args.gguf_extents_path} missing; "
                      "run 'xnvme-driver reset' and retry")
            return 1
        dev = _resolve_block_device(cijoe, pci_addr, args.dma_namespace)
        if not dev:
            return 1
        log.info(f"DMA target: {pci_addr} -> {dev} (layout=xfs-fiemap)")
        return _setup_xfs_fiemap(args, cijoe, pci_addr, dev)

    return _setup_raw(args, cijoe, pci_addr)
