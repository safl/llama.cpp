"""Point the target's xNVMe tree at a specific fork+branch and rebuild.

Used to iterate on xNVMe changes (e.g. the upcie-cuda mem_map wiring)
without waiting for them to land in upstream main. Ensures the given
`remote_name` maps to `remote_url`, fetches, checks out `branch`,
wipes the builddir, and reinstalls.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--xnvme_src", type=str, default="/root/xnvme")
    parser.add_argument("--remote_name", type=str, default="safl")
    parser.add_argument("--remote_url", type=str,
                        default="https://github.com/safl/xNVMe.git")
    parser.add_argument("--branch", type=str, required=True)


def main(args, cijoe):
    err, _ = cijoe.run(f'[ -d "{args.xnvme_src}/.git" ]')
    if err:
        # Not a git tree - blow it away and re-clone the branch.
        log.info(f"{args.xnvme_src} is not a git tree; wiping and cloning fresh")
        err, _ = cijoe.run(f"rm -rf {args.xnvme_src}")
        if err:
            log.error(f"rm -rf {args.xnvme_src} failed")
            return err
        err, _ = cijoe.run(
            f"git clone --recurse-submodules --shallow-submodules -b {args.branch} "
            f"{args.remote_url} {args.xnvme_src}"
        )
        if err:
            log.error(f"git clone {args.remote_url}#{args.branch} failed")
            return err
    else:
        # Add/refresh remote
        err, _ = cijoe.run(f"git -C {args.xnvme_src} remote get-url {args.remote_name} 2>/dev/null")
        if err:
            err, _ = cijoe.run(f"git -C {args.xnvme_src} remote add {args.remote_name} {args.remote_url}")
            if err:
                log.error(f"could not add remote {args.remote_name} -> {args.remote_url}")
                return err
        else:
            cijoe.run(f"git -C {args.xnvme_src} remote set-url {args.remote_name} {args.remote_url}")

        err, _ = cijoe.run(f"git -C {args.xnvme_src} fetch {args.remote_name} {args.branch}")
        if err:
            log.error(f"fetch {args.remote_name}/{args.branch} failed")
            return err

        err, _ = cijoe.run(f"git -C {args.xnvme_src} checkout -B {args.branch} {args.remote_name}/{args.branch}")
        if err:
            log.error(f"checkout {args.branch} failed")
            return err

    err, _ = cijoe.run(f"git -C {args.xnvme_src} submodule update --init --recursive")
    if err:
        log.error("submodule update failed")
        return err

    cijoe.run(f"rm -rf {args.xnvme_src}/builddir")

    for cmd in ["make config", "make build", "make install", "ldconfig"]:
        err, _ = cijoe.run(cmd, cwd=args.xnvme_src)
        if err:
            log.error(f"'{cmd}' failed in {args.xnvme_src}")
            return err

    return 0
