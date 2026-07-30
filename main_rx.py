#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import adi
import numpy as np
from cryptography.fernet import InvalidToken

from cosmos import PlutoReceiver
from digicomm import qam_symbols_to_bits
from helpers import bits_to_bytes, decompress_file, decrypt_file


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

#Change only this number to control the length of each completed log, in sec
ROTATION_SECONDS = 3600
OUTPUT_DIR = Path("/Users/vincent/Desktop/SDRReceivedLogs")
#Change the PLUTO URI everytime u use it
PLUTO_URI = "usb:1.1.5"
CHANNEL = 7
GAIN_LEVEL = 80
RX_BUFFER_SIZE = int(1e6)


M = 16
HEADER_BITS = 32
MAX_BITS = 19968

KEY = b"PatTEws1o7HD5TpT-9IowWCdhxXvOKFXsQJxoAWf_lQ="


def utc_file_timestamp(value: datetime) -> str:
    """Format a UTC timestamp for a completed log filename."""
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

#创建临时 .part 文件
#        ↓
#不断 append 新消息
#        ↓
#ROTATION_SECONDS 到期
#        ↓
#关闭并保存为正式 .log
#        ↓
#马上创建新的 .part
#        ↓
#继续循环
class RotatingLogWriter:
    """Collect received payloads in a temporary file and rotate by time."""

    def __init__(self, output_dir: Path, rotation_seconds: int) -> None:
        if rotation_seconds <= 0:
            raise ValueError("ROTATION_SECONDS must be greater than zero")

        self.output_dir = output_dir
        self.rotation_seconds = rotation_seconds
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # The .part extension prevents watcher.py from treating the active
        # window as a completed log.
        self.active_path = self.output_dir / f".active_sdr_{os.getpid()}.part"
        self.active_file = None
        self.window_start = datetime.now(timezone.utc)
        self.window_end = self.window_start
        self.deadline = time.monotonic()
        self.has_data = False
        self._open_window(self.window_start)

    def _open_window(self, start: datetime) -> None:
        self.window_start = start
        self.window_end = start + timedelta(seconds=self.rotation_seconds)
        self.deadline = time.monotonic() + self.rotation_seconds
        self.has_data = False
        self.active_file = self.active_path.open("wb")
        print(
            "Opened active log window: "
            f"{utc_file_timestamp(self.window_start)} to "
            f"{utc_file_timestamp(self.window_end)}"
        )

    def _completed_path(self, end: datetime) -> Path:
        filename = (
            f"messages_{utc_file_timestamp(self.window_start)}"
            f"--{utc_file_timestamp(end)}.log"
        )
        path = self.output_dir / filename

        # Never overwrite a completed file, even in the extremely unlikely
        # event of an identical timestamp.
        if path.exists():
            filename = (
                f"messages_{utc_file_timestamp(self.window_start)}"
                f"--{utc_file_timestamp(end)}"
                f"_{time.time_ns()}.log"
            )
            path = self.output_dir / filename
        return path

    def _finish_window(self, end: datetime) -> None:
        if self.active_file is None:
            return

        self.active_file.flush()
        os.fsync(self.active_file.fileno())
        self.active_file.close()
        self.active_file = None

        if self.has_data:
            completed_path = self._completed_path(end)
            self.active_path.replace(completed_path)
            print(f"Completed log sent to: {completed_path}")
        else:
            # Do not send an empty log to the summarizer.
            self.active_path.unlink(missing_ok=True)
            print("Window contained no decoded messages; no log was sent.")

    def rotate_if_due(self) -> None:
        """Finalize the current window and immediately create the next one."""
        if time.monotonic() < self.deadline:
            return

        scheduled_end = self.window_end
        self._finish_window(scheduled_end)
        self._open_window(scheduled_end)

    def append(self, payload: bytes) -> None:
        """Append one successfully decoded .log payload to the active window."""
        self.rotate_if_due()
        if self.active_file is None:
            raise RuntimeError("The active log is not open")

        # Keep separate received payloads readable without altering their
        # internal Telegram-message formatting.
        if self.has_data:
            self.active_file.write(b"\n")
        self.active_file.write(payload)
        if payload and not payload.endswith(b"\n"):
            self.active_file.write(b"\n")
        self.active_file.flush()
        self.has_data = True

    def close(self) -> None:
        """On Control-C, preserve the current partial window."""
        if self.active_file is not None:
            self._finish_window(datetime.now(timezone.utc))

#Keep only /who, /platform, and /message from each received message.
def select_message_fields(payload: bytes) -> bytes:
    text = payload.decode("utf-8")
    selected_entries: list[str] = []
    current_entry: dict[str, str] = {}

    def finish_entry() -> None:
        if all(field in current_entry for field in ("who", "platform", "message")):
            selected_entries.append(
                "\n".join(
                    (
                        current_entry["who"],
                        current_entry["platform"],
                        current_entry["message"],
                    )
                )
            )
        current_entry.clear()

    for line in text.splitlines():
        if line.startswith("/who "):
            # A new /who line begins a new message entry.
            if current_entry:
                finish_entry()
            current_entry["who"] = line
        elif line.startswith("/platform "):
            current_entry["platform"] = line
        elif line.startswith("/message "):
            current_entry["message"] = line
            finish_entry()

    if current_entry:
        finish_entry()

    if not selected_entries:
        raise ValueError(
            "Decoded payload contains no complete "
            "/who, /platform, /message entries"
        )

    return ("\n\n".join(selected_entries) + "\n").encode("utf-8")






def main() -> None:
    # ---------------------------------------------------------------
    # Setup Pluto receiver
    # ---------------------------------------------------------------
    sdr_rx = adi.Pluto(PLUTO_URI)
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_buffer_size(RX_BUFFER_SIZE)
    rx.set_channel(CHANNEL)
    rx.set_gain_level(GAIN_LEVEL)
    rx.desired_transmit_symbols_real = False
    bits_per_symbol = int(np.log2(M))
    rx.num_transmit_symbols = (HEADER_BITS + MAX_BITS) // bits_per_symbol
    # Create the temporary rotating log.
    writer = RotatingLogWriter(OUTPUT_DIR, ROTATION_SECONDS)

                               
    print("Continuous rotating receiver started.")
    print(f"Completed logs will be saved to: {OUTPUT_DIR}")
    print(f"Rotation interval: {ROTATION_SECONDS} seconds")
    print("Press Control-C to stop.")


    try:
        while True:
            writer.rotate_if_due()
            try:
                rx_symbols = rx.receive()
                rx_bits = qam_symbols_to_bits(rx_symbols, M, 0)

                # Make sure the 32-bit header exists.
                if len(rx_bits) < HEADER_BITS:
                    raise ValueError("Frame is too short for its header: "f"{len(rx_bits)} bits")

                header_bit_values = rx_bits[:HEADER_BITS]
                header_string = "".join(str(int(bit))for bit in header_bit_values)
                message_len = int(header_string, 2)

                # Validate the payload length.
                if message_len <= 0 or message_len > MAX_BITS:
                    raise ValueError(f"Invalid message length: {message_len}")
                if message_len % 8 != 0:
                    raise ValueError("Message length is not byte-aligned: "f"{message_len}")

                required_bits = HEADER_BITS + message_len

                if len(rx_bits) < required_bits:
                    raise ValueError(f"Incomplete frame: need {required_bits} bits, "f"received {len(rx_bits)}")

                message_bits = rx_bits[HEADER_BITS:required_bits]
                rx_bytes = bits_to_bytes(message_bits)
                rx_bytes = decrypt_file(rx_bytes, KEY)
                rx_bytes = decompress_file(rx_bytes)
                selected_bytes = select_message_fields(rx_bytes)
                # add the selected messages to the temporary .log
                writer.append(selected_bytes)
                message_count = selected_bytes.count(b"/message ")
                print(f"Added {message_count} received message(s) to the active log.")

            except (InvalidToken, ValueError) as error:
                print(f"Decode failed: {error}")
            except Exception as error:
                print(f"Receive failed: {error}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")

    finally:
        # Save current partial window when Control-C is pressed
        writer.close()
        print("Continuous rotating receiver stopped!")

if __name__ == "__main__":
    main()
