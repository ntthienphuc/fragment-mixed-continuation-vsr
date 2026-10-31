"""Record server failure for the cost guard without changing the sealed scientific entry."""

import argparse
import subprocess
import sys

from common import HERE, save, utc


def main(deadline):
    code = subprocess.call([sys.executable, "-B", str(HERE / "server_entry.py"),
                            "--deadline", str(deadline)])
    if code:
        save(HERE / "Server_Failure.json", {"utc": utc(), "exit_code": code})
    raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args().deadline)
