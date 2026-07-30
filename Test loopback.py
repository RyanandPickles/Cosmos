#!/usr/bin/env python3
"""
Loopback test — no Pluto SDR hardware needed.

Watches the Telegram logger's logs/ directory for new .log files,
runs each one through the full TX pipeline (compress → encrypt → QAM encode)
and immediately back through the full RX pipeline (QAM decode → decrypt →
decompress), then writes the recovered file to the output folder so you can
confirm the content is identical to what went in.

Usage:
    cd /Users/ryanli/Cosmos
    python test_loopback.py

Press Ctrl-C to stop.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from cryptography.fernet import Fernet, InvalidToken

# ── path setup so helpers/digicomm/cosmos import correctly ────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    file_to_bytes, bytes_to_file,
    bytes_to_bits, bits_to_bytes,
    compress_file, decompress_file,
    encrypt_file, decrypt_file,
)
from digicomm import bits_to_qam_symbols, qam_symbols_to_bits, get_qam_constellation

# ── config ────────────────────────────────────────────────────
LOGS_DIR   = Path("/Users/ryanli/Cosmos/Telegram/telegram-message-logger/logs")
OUTPUT_DIR = Path("/Users/ryanli/Desktop/untitled folder")
KEY        = b"PatTEws1o7HD5TpT-9IowWCdhxXvOKFXsQJxoAWf_lQ="

M           = 16
HEADER_BITS = 32
MAX_BITS    = 19968

# ── pipeline ──────────────────────────────────────────────────

def encode(file_path: Path) -> tuple[np.ndarray, int]:
    """TX side: file → compress → encrypt → bits → QAM symbols."""
    raw        = file_to_bytes(str(file_path))
    compressed = compress_file(raw)
    encrypted  = encrypt_file(compressed, KEY)
    bits       = bytes_to_bits(encrypted)

    if len(bits) > MAX_BITS:
        raise ValueError(f"File too large after compress+encrypt: {len(bits)} bits > {MAX_BITS}")

    message_len    = len(bits)
    header_bits    = np.array([int(b) for b in format(message_len, f'0{HEADER_BITS}b')])
    padding_zeros  = MAX_BITS - message_len
    pad_bits       = np.random.randint(0, 2, padding_zeros)
    all_bits       = np.concatenate((header_bits, bits, pad_bits))

    symbols, remainder = bits_to_qam_symbols(all_bits, M)
    return symbols, remainder


def decode(symbols: np.ndarray, remainder: int) -> bytes:
    """RX side: QAM symbols → bits → decrypt → decompress → bytes."""
    rx_bits = qam_symbols_to_bits(symbols, M, remainder)

    header_string = "".join(str(b) for b in rx_bits[:HEADER_BITS])
    message_len   = int(header_string, 2)

    if message_len <= 0 or message_len > MAX_BITS:
        raise ValueError(f"Bad header — decoded message_len={message_len}")

    message_bits = rx_bits[HEADER_BITS: HEADER_BITS + message_len]
    rx_bytes     = bits_to_bytes(message_bits)
    decrypted    = decrypt_file(rx_bytes, KEY)
    return decompress_file(decrypted)


def process(log_path: Path) -> None:
    """Run one log file through the full loopback and write result."""
    print(f"\n{'─'*60}")
    print(f"[test] Processing: {log_path.name}")

    original_bytes = file_to_bytes(str(log_path))
    if not original_bytes.strip():
        print("[test] Empty log file — skipping.")
        return

    # ── TX ──
    try:
        symbols, remainder = encode(log_path)
        print(f"[TX]   compress+encrypt+QAM → {len(symbols)} symbols")
    except ValueError as e:
        print(f"[TX]   FAILED: {e}")
        return

    # ── loopback: symbols go straight to RX (no RF) ──
    try:
        recovered = decode(symbols, remainder)
    except (InvalidToken, ValueError, Exception) as e:
        print(f"[RX]   FAILED: {e}")
        return

    # ── verify ──
    match = recovered == original_bytes
    print(f"[RX]   decode+decrypt+decompress → {len(recovered)} bytes")
    print(f"[✓]   Content match: {match}")
    if not match:
        print(f"      Original  : {original_bytes[:120]}")
        print(f"      Recovered : {recovered[:120]}")

    # ── write output ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path    = OUTPUT_DIR / f"loopback_{stamp}_{log_path.stem}.log"
    bytes_to_file(recovered, str(out_path))
    print(f"[out]  Saved → {out_path}")


# ── watcher ───────────────────────────────────────────────────

def watch() -> None:
    if not LOGS_DIR.exists():
        print(f"[test] Logs dir not found: {LOGS_DIR}")
        print("[test] Start the Telegram logger first, then re-run this script.")
        sys.exit(1)

    print(f"[test] Watching: {LOGS_DIR}")
    print(f"[test] Output:   {OUTPUT_DIR}")
    print("[test] Press Ctrl-C to stop.\n")

    handled: set[Path] = set()

    # mark anything already in the folder as seen so we only process new files
    for f in sorted(LOGS_DIR.glob("*.log")):
        handled.add(f)
    print(f"[test] Skipping {len(handled)} existing log(s), waiting for new ones...")

    try:
        while True:
            current = sorted(LOGS_DIR.glob("*.log"))
            new     = [f for f in current if f not in handled]

            for log_file in new:
                # wait a beat for the file to finish being written
                time.sleep(0.5)
                handled.add(log_file)
                try:
                    process(log_file)
                except Exception as e:
                    print(f"[test] Unexpected error on {log_file.name}: {e}")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[test] Stopped.")


if __name__ == "__main__":
    watch()