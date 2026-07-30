#!/usr/bin/env python3
"""Watch for completed Telegram log windows and run transfer.py."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
TRANSFER_SCRIPT = BASE_DIR / "transfer.py"
POLL_SECONDS = 0.10
SETTLE_SECONDS = 0.25

# timed-window files:  messages_<start>--<end>.log  (old rotation mode)
LOG_WINDOWED = re.compile(
    r"^messages_\d{8}T\d{6}\.\d{6}Z"
    r"--(?P<end>\d{8}T\d{6}\.\d{6}Z)\.log$"
)
# per-message files:   messages_<timestamp>.log  (new per-message mode)
LOG_SINGLE = re.compile(
    r"^messages_\d{8}T\d{6}\.\d{6}Z\.log$"
)


def load_log_directory() -> Path:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    log_directory = Path(str(config.get("log_directory", "logs"))).expanduser()
    if not log_directory.is_absolute():
        log_directory = BASE_DIR / log_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    return log_directory


def is_ready(path: Path, now: datetime) -> bool:
    """Return True if this file is complete and safe to send.

    Per-message files (LOG_SINGLE) are written atomically by PerMessageLogWriter
    so they're ready the moment they appear — no window to wait for.
    Timed-window files (LOG_WINDOWED) are ready once their end timestamp has
    passed plus a small settle buffer.
    """
    if LOG_SINGLE.fullmatch(path.name):
        return True
    match = LOG_WINDOWED.fullmatch(path.name)
    if match:
        end = datetime.strptime(match.group("end"), "%Y%m%dT%H%M%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        return now.timestamp() >= end.timestamp() + SETTLE_SECONDS
    return False


def run_transfer(path: Path) -> None:
    if not TRANSFER_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing transfer program: {TRANSFER_SCRIPT}")
    subprocess.run(
        [sys.executable, str(TRANSFER_SCRIPT), str(path.resolve())],
        cwd=BASE_DIR,
        check=True,
    )


def main() -> None:
    log_directory = load_log_directory()
    already_handled = {
        path.resolve()
        for path in log_directory.glob("messages_*.log")
        if is_ready(path, datetime.now(timezone.utc))
    }

    print(f"Watching for completed logs in {log_directory}", flush=True)
    print("Press Control-C to stop the watcher.", flush=True)

    try:
        while True:
            now = datetime.now(timezone.utc)
            for path in sorted(log_directory.glob("messages_*.log")):
                resolved = path.resolve()
                if resolved in already_handled or not is_ready(path, now):
                    continue

                try:
                    run_transfer(resolved)
                except Exception as exc:
                    print(f"Transfer failed for {resolved}: {exc}", file=sys.stderr)
                    continue

                already_handled.add(resolved)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nWatcher stopped.", flush=True)


if __name__ == "__main__":
    main()
