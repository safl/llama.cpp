"""
Build llama.cpp (with CUDA) and the xnvme-load benchmark tool on the remote
host. Assumes git_sync has already placed the source at ``llama_src`` on the
remote.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--llama_src", type=str, required=True)
    parser.add_argument("--llama_build", type=str, required=True)
    parser.add_argument("--cuda_arch", type=str, default="86")
    parser.add_argument("--jobs", type=int, default=0)


def main(args, cijoe):
    jobs = f"-j{args.jobs}" if args.jobs else "-j"

    err, _ = cijoe.run(
        f"cmake -S {args.llama_src} -B {args.llama_build} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DGGML_CUDA=ON "
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch} "
        f"-DLLAMA_BUILD_TESTS=OFF "
        f"-DLLAMA_BUILD_EXAMPLES=OFF"
    )
    if err:
        log.error("cmake configure failed")
        return err

    # Build llama-completion (one-shot, tty-free) instead of llama-cli
    # (which was refactored into a chat-only REPL). run_baseline runs
    # the resulting binary directly out of build/bin so we skip cmake
    # --install.
    err, _ = cijoe.run(
        f"cmake --build {args.llama_build} {jobs} -t llama-completion"
    )
    if err:
        log.error("cmake build failed")
        return err

    xnvme_load_src = f"{args.llama_src}/cijoe/tools/xnvme-load"
    xnvme_load_build = f"{xnvme_load_src}/build"

    err, _ = cijoe.run(f"meson setup --wipe {xnvme_load_build} {xnvme_load_src}")
    if err:
        log.error("meson setup failed for xnvme-load")
        return err

    err, _ = cijoe.run(f"meson install -C {xnvme_load_build}")
    if err:
        log.error("meson install failed for xnvme-load")
        return err

    return 0
