"""Aggressively free the GPU: kill llama-*, kill MPS, reset compute mode, print memory."""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("pkill -9 -f llama-cli    || true")
    cijoe.run("pkill -9 -f llama-completion || true")
    cijoe.run("pkill -9 -f llama-server || true")
    cijoe.run("echo quit | nvidia-cuda-mps-control 2>/dev/null || true")
    cijoe.run("pkill -9 -f nvidia-cuda-mps || true")
    cijoe.run("sleep 2")
    cijoe.run("nvidia-smi -c DEFAULT")
    cijoe.run("nvidia-smi --gpu-reset -i 0 2>&1 || true")
    cijoe.run("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv")
    cijoe.run("fuser -v /dev/nvidia* 2>&1 || true")
    return 0
