"""Print GPU processes and PID mapping details."""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    cijoe.run("nvidia-smi")
    cijoe.run("lsof /dev/nvidia* 2>&1 | head -30")
    cijoe.run("cat /proc/driver/nvidia/gpus/*/information 2>&1 | head -20")
    return 0
