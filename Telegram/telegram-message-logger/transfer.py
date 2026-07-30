from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MAIN_TX = BASE_DIR.parent.parent / "main_tx.py"
COSMOS_PYTHON = "python"


def transfer_log(log_path: Path) -> None:
    subprocess.run(
        [COSMOS_PYTHON, str(MAIN_TX), str(log_path)],
        cwd=BASE_DIR,
        check=True,
    )
    log_path.unlink()

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            f"Usage: {Path(sys.argv[0]).name} /path/to/file.log"
        )

    log_path = Path(sys.argv[1]).expanduser().resolve()

    if not log_path.is_file():
        raise SystemExit(f"File not found: {log_path}")

    transfer_log(log_path)


if __name__ == "__main__":
    main()
