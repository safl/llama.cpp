"""One-shot: dump current driver bindings for each PCI NVMe on the target."""

from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--pci_addrs", type=str, default="0000:01:00.0,0000:c1:00.0,0000:c2:00.0,0000:c3:00.0,0000:c4:00.0")


def main(args, cijoe):
    for addr in [a.strip() for a in args.pci_addrs.split(",") if a.strip()]:
        cijoe.run(f"echo '=== {addr} ==='")
        cijoe.run(f"lspci -k -s {addr}")
        cijoe.run(f"ls /sys/bus/pci/devices/{addr}/nvme/ 2>/dev/null || echo '(no nvme controller)'")
    return 0
