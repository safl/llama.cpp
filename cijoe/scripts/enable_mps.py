"""Enable CUDA Multi-Process Service (MPS) on the target.

Multi-process CUDA on a single GPU without MPS causes the driver
to time-slice between contexts, which serialises the four
llama.cpp instances in the parallel benchmark. Enabling MPS lets
them share the GPU concurrently.

Requires root; sets EXCLUSIVE_PROCESS compute mode + starts
the MPS control daemon. Idempotent: kills any prior daemon and
respawns cleanly. Prints the current state so we can eyeball it
in cijoe output.
"""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("echo quit | nvidia-cuda-mps-control 2>/dev/null || true")
    cijoe.run("pkill -f nvidia-cuda-mps || true")
    cijoe.run("sleep 1")
    err, _ = cijoe.run("nvidia-smi -c EXCLUSIVE_PROCESS")
    if err:
        return err
    err, _ = cijoe.run("nvidia-cuda-mps-control -d")
    if err:
        return err
    cijoe.run("sleep 1")
    cijoe.run("echo get_server_list | nvidia-cuda-mps-control")
    cijoe.run("nvidia-smi -q -d COMPUTE | head -20")
    return 0
