"""
In a development environment where the project source is edited on a "local"
system, and deployed for build and test on another/different "remote" system.

Then one needs to transfer the source, or changes to the source, to the remote
system. This can be done with tools such as rsync and mutagen. However, these
can cause unwanted behavior when syncing repository data such as ``.git``.

Ideally, one would execute the required git commands, transferring the changes
to the remote, however, this can be tedious. Thus, this script, running the
various commands, is doing just that.

Here is what it does:

* remote: checkout main
* local: commit changes with message "autocommit ..."
* local: push changes to remote
* remote: checkout step.repository.branch

With the above, then git is utilized for syncing, and the cijoe-scripting takes
care of the need to switch branch remotely.
"""

import logging as log
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--upstream", type=str, required=True)
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--remote_alias", type=str, required=True)
    parser.add_argument("--local_path", type=str, required=True)
    parser.add_argument("--remote_path", type=str, required=True)
    parser.add_argument("--base_branch", type=str, default="main",
                        help="branch to check out on remote before pushing "
                             "(must differ from --branch since git refuses to "
                             "push to the currently-checked-out branch of a "
                             "non-bare repo)")


def git_remote_from_config(cijoe, remote_path):
    """Returns git-remote URI and SSH config using configuration cijoe.transport.ssh"""

    hostname = cijoe.getconf("cijoe.transport.ssh.hostname", None)
    if not hostname:
        return None, {}

    username = cijoe.getconf("cijoe.transport.ssh.username", None)
    port = cijoe.getconf("cijoe.transport.ssh.port", None)
    password = cijoe.getconf("cijoe.transport.ssh.password", None)

    remote = "ssh://"
    if username:
        remote += f"{username}@"
    remote += f"{hostname}"
    if port:
        remote += f":{port}"
    remote += f"{remote_path}"

    ssh_conf = {"password": password, "port": port}

    return remote, ssh_conf


def main(args, cijoe):
    """Entry point"""

    if any(
        arg not in args
        for arg in ["upstream", "branch", "remote_alias", "local_path", "remote_path"]
    ):
        log.error("missing script arguments")

    local_path = args.local_path

    upstream = args.upstream
    branch = args.branch

    remote_path = args.remote_path
    remote_alias = args.remote_alias
    remote_url, ssh_conf = git_remote_from_config(cijoe, remote_path)
    if not remote_url:
        return 1

    # Build GIT_SSH_COMMAND so git push uses the config credentials
    ssh_cmd_parts = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if ssh_conf.get("port"):
        ssh_cmd_parts += ["-p", str(ssh_conf["port"])]
    ssh_cmd = " ".join(ssh_cmd_parts)
    if ssh_conf.get("password"):
        password = ssh_conf["password"]
        push_prefix = f'SSHPASS="{password}" GIT_SSH_COMMAND="sshpass -e {ssh_cmd}"'
    else:
        push_prefix = f'GIT_SSH_COMMAND="{ssh_cmd}"'

    # Remotely: clone repository from upstream
    err, _ = cijoe.run(f'[ -d "{remote_path}/.git" ]')
    if err:
        for cmd in [
            f"mkdir -p {remote_path}",
            f"git clone {upstream} {remote_path}",
        ]:
            err, _ = cijoe.run(cmd)
            if err:
                return err

    # Locally: ensure that we have the remote alias set up to push to.
    # git remote -v uses TAB between name and URL, not space; state.output()
    # yields the raw string not a per-line list, so split explicitly.
    err, state = cijoe.run_local("git remote -v")
    if err:
        return err
    out = state.output()
    lines = out.splitlines() if isinstance(out, str) else out
    have_alias = any(
        line.startswith(f"{remote_alias}\t") and line.endswith(" (push)")
        for line in lines
    )
    if not have_alias:
        err, _ = cijoe.run_local(f"git remote add {remote_alias} {remote_url}")
        if err:
            return err

    # Remotely: check out the base branch so 'branch' is free to be pushed
    err, _ = cijoe.run(f"git checkout {args.base_branch}", cwd=remote_path)
    if err:
        return err
    # Locally: autocommit anything unstaged (harmless no-op if clean)
    err, state = cijoe.run_local(
        'git commit -a -s -m "autocommit ..."', cwd=local_path
    )
    if err:
        no_change_markers = (
            "nothing to commit",
            "nothing added to commit",
            "no changes added to commit",
        )
        if not any(m in state.output() for m in no_change_markers):
            return err

    # Locally: push to remote using the ssh credentials from cijoe config.
    # There is only ever one push; the previous shape had a preliminary
    # push without the ssh-auth wrapper which failed on password-auth
    # targets that lack ssh-askpass.
    err, state = cijoe.run_local(
        f"{push_prefix} git push {remote_alias} HEAD:{branch} -f", cwd=local_path
    )
    if err:
        return err

    err, _ = cijoe.run(f"git checkout {branch}", cwd=remote_path)
    if err:
        return err
