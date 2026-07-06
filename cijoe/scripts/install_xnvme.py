"""
Install xNVMe on the target host if pkg-config cannot already find it.

Idempotent. Skip everything when xnvme is already discoverable via
pkg-config and reports XNVME_BE_LINUX_LIBURING_ENABLED. Otherwise
shallow-clone xnvme (or reuse an existing tree at ``xnvme_src``),
install its build-time deps via the tree's own
``toolbox/pkgs/<distro>.sh``, and drive its canonical
``make config`` / ``make build`` / ``make install`` cycle.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--xnvme_src", type=str, default="/root/xnvme")
    parser.add_argument("--xnvme_repo", type=str,
                        default="https://github.com/xnvme/xnvme.git")
    parser.add_argument("--xnvme_branch", type=str, default="main")


def _install_pkgs(cijoe, xnvme_src):
    """Install xnvme's build-time deps using the tree's own
    toolbox/pkgs/<distro>-<codename>.sh. Falls back to liburing-dev
    only if there is no matching script."""
    return cijoe.run(
        'set -e; '
        '. /etc/os-release; '
        f'pkg_script="{xnvme_src}/toolbox/pkgs/${{ID}}-${{VERSION_CODENAME}}.sh"; '
        'if [ -x "$pkg_script" ]; then '
        '  echo "running $pkg_script"; bash "$pkg_script"; '
        'else '
        '  echo "no matching pkgs script for ${ID}-${VERSION_CODENAME}, '
        'falling back to liburing-dev"; '
        '  apt-get install -y liburing-dev; '
        'fi'
    )


def main(args, cijoe):
    err, _ = cijoe.run(f'[ -f "{args.xnvme_src}/Makefile" ]')
    if err:
        log.info(f"no xnvme tree at {args.xnvme_src}, cloning")
        err, _ = cijoe.run(
            f"git clone --recurse-submodules --shallow-submodules "
            f"--depth 1 -b {args.xnvme_branch} "
            f"{args.xnvme_repo} {args.xnvme_src}"
        )
        if err:
            log.error("git clone of xnvme failed")
            return err
    else:
        log.info(f"reusing existing xnvme tree at {args.xnvme_src}")

    err, _ = _install_pkgs(cijoe, args.xnvme_src)
    if err:
        log.error("failed to install xnvme build-time packages")
        return err

    err, _ = cijoe.run("pkg-config --exists xnvme")
    if not err:
        err_iou, _ = cijoe.run(
            "xnvme library-info 2>&1 | grep -q XNVME_BE_LINUX_LIBURING_ENABLED"
        )
        if not err_iou:
            log.info("xnvme + liburing already installed, skipping")
            return 0
        log.info("xnvme installed but without liburing; wiping builddir "
                 "so it reconfigures")
        cijoe.run(f"rm -rf {args.xnvme_src}/builddir")

    err, _ = cijoe.run(f'[ -d "{args.xnvme_src}/builddir" ]')
    if err:
        for cmd in ["make config", "make build"]:
            err, _ = cijoe.run(cmd, cwd=args.xnvme_src)
            if err:
                log.error(f"'{cmd}' failed in {args.xnvme_src}")
                return err
    else:
        log.info(f"reusing existing builddir at {args.xnvme_src}/builddir")

    for cmd in ["make install", "ldconfig"]:
        err, _ = cijoe.run(cmd, cwd=args.xnvme_src)
        if err:
            log.error(f"'{cmd}' failed in {args.xnvme_src}")
            return err

    err, _ = cijoe.run("pkg-config --exists xnvme")
    if err:
        log.error("xnvme install completed but pkg-config still cannot find it")
        return 1

    return 0
