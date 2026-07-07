"""One-shot: kill any lingering llama-cli / llama-completion / llama-server on the target and dump current GPU memory."""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("pkill -f llama-cli || true")
    cijoe.run("pkill -f llama-completion || true")
    cijoe.run("pkill -f llama-server || true")
    cijoe.run("sleep 1 && nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv")
    return 0
