"""Run `xnvme-driver reset` to re-bind any uio_pci_generic-detached drives back to the kernel nvme driver."""


def add_args(parser):
    pass


def main(args, cijoe):
    err, _ = cijoe.run("xnvme-driver reset")
    return err
