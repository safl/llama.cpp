"""Print state of /root/xnvme on the target so we can figure out how to update it."""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("ls -la /root/xnvme | head -20")
    cijoe.run("[ -d /root/xnvme/.git ] && echo IS_GIT || echo NOT_GIT")
    cijoe.run("cat /root/xnvme/VERSION 2>/dev/null || cat /root/xnvme/toolbox/third-party/linux/upcie/cudamem_mapping.h 2>/dev/null | head -3 || echo NO_MARKERS")
    cijoe.run("grep -l cudamem_mapping /root/xnvme/toolbox/third-party/linux/upcie/*.h 2>/dev/null || echo 'no cudamem_mapping.h'")
    cijoe.run("head -5 /root/xnvme/lib/upcie/xnvme_be_upcie_cuda_mem.c 2>/dev/null; grep -n 'mem_map' /root/xnvme/lib/upcie/xnvme_be_upcie_cuda_mem.c 2>/dev/null")
    return 0
