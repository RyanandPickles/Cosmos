#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import adi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cosmos import PlutoTransmitter
from main_tx import transmit_file

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
POLL_SECONDS = 0.10
SETTLE_SECONDS = 0.25

PLUTO_URI = "usb:0.1.5"
CHANNEL = 7
POWER_LEVEL = 90

LOG_WINDOWED = re.compile(
    r"^messages_\d{8}T\d{6}\.\d{6}Z"
    r"--(?P<end>\d{8}T\d{6}\.\d{6}Z)\.log$"
)
LOG_SINGLE = re.compile(
    r"^messages_\d{8}T\d{6}\.\d{6}Z\.log$"
)


def load_log_directory():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    log_directory = Path(str(config.get("log_directory", "logs"))).expanduser()
    if not log_directory.is_absolute():
        log_directory = BASE_DIR / log_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    return log_directory


def is_ready(path, now):
    if LOG_SINGLE.fullmatch(path.name):
        return True
    match = LOG_WINDOWED.fullmatch(path.name)
    if match:
        end = datetime.strptime(match.group("end"), "%Y%m%dT%H%M%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        return now.timestamp() >= end.timestamp() + SETTLE_SECONDS
    return False


def main():
    print("Initializing Pluto SDR...", flush=True)
    sdr_tx = adi.Pluto(PLUTO_URI)
    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_channel(CHANNEL)
    tx.set_power_level(POWER_LEVEL)
    print("Pluto ready.", flush=True)

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
                    transmit_file(str(resolved), tx, sleep_seconds=1)
                except Exception as exc:
                    print(f"Transfer failed for {resolved}: {exc}", file=sys.stderr)
                already_handled.add(resolved)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nWatcher stopped.", flush=True)


if __name__ == "__main__":
    main()