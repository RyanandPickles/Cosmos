"""Not used by watcher.py in the current setup — kept for reference only.
watcher.py now calls transmit_file() directly instead of spawning this as a subprocess."""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MAIN_TX = BASE_DIR.parent.parent / "main_tx.py"

COSMOS_PYTHON = "python3"


def transfer_log(log_path: Path) -> None:
    subprocess.run(
        [COSMOS_PYTHON, str(MAIN_TX), str(log_path)],
        check=True,
    )


if __name__ == "__main__":
    transfer_log(Path(sys.argv[1]))