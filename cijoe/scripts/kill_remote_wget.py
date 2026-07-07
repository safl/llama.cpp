"""One-shot: kill any lingering wget on the remote host."""


def add_args(parser):
    pass


def main(args, cijoe):
    cijoe.run("pkill -f wget || true")
    cijoe.run("sleep 1 && pgrep -a wget || echo 'no wget'")
    return 0
