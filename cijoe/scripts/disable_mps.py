"""Turn CUDA MPS off and reset compute mode to DEFAULT so subsequent processes see the whole VRAM."""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("echo quit | nvidia-cuda-mps-control 2>/dev/null || true")
    cijoe.run("pkill -f nvidia-cuda-mps || true")
    cijoe.run("sleep 1")
    cijoe.run("nvidia-smi -c DEFAULT")
    cijoe.run("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv")
    return 0
