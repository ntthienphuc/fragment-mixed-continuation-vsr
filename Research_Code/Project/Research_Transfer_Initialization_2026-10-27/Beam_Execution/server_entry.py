"""Sequential cross-device checks followed by the finite disjoint server queue."""

import argparse
import subprocess
import sys
import time

from common import HERE, plan


def main(deadline):
    for model in plan()["models"]:
        assert time.time() < deadline
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "server_preflight.py"), model], check=True)
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(HERE / "task_queue.py"),
                    "--owner", "server", "--attempt", "Server_01", "--parallel", "2",
                    "--deadline", str(deadline)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, required=True)
    main(parser.parse_args().deadline)
