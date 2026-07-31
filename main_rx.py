#!/usr/bin/env python3
from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import adi
import numpy as np
from cryptography.fernet import InvalidToken
from cosmos import PlutoReceiver
from digicomm import qam_symbols_to_bits
from helpers import *
import traceback

OUTPUT_DIR = Path("/Users/vincent/Desktop/SDRReceivedLogs")
PLUTO_URI = "usb:0.1.5"
CHANNEL = 7
GAIN_LEVEL = 80
RX_BUFFER_SIZE = int(1e6)
ROTATION_SECONDS = 60

M = 16
HEADER_BITS = 32
MAX_BITS = 7968


def main() -> None:
    sdr_rx = adi.Pluto(PLUTO_URI)
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_buffer_size(RX_BUFFER_SIZE)
    rx.set_channel(CHANNEL)
    rx.set_gain_level(GAIN_LEVEL)
    rx.desired_transmit_symbols_real = False
    bits_per_symbol = int(np.log2(M))
    rx.num_transmit_symbols = (HEADER_BITS + MAX_BITS) // bits_per_symbol

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Receiver started. Waiting for transmissions...")
    print(f"Saving to: {OUTPUT_DIR}")
    print("Press Control-C to stop.")

    window_start = time.monotonic()
    window_messages = []
    last_message = None
    try:
        while True:
            if time.monotonic() - window_start >= ROTATION_SECONDS:
                if window_messages:
                    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    output_path = OUTPUT_DIR / f"rx_{timestamp}.log"
                    output_path.write_bytes(b"\n".join(window_messages))
                    print(f"Saved {len(window_messages)} messages to {output_path}")
                window_messages = []
                window_start = time.monotonic()

            try:
                rx_symbols = rx.receive()
                rx_bits = qam_symbols_to_bits(rx_symbols, M, 0)

                if len(rx_bits) < HEADER_BITS:
                    print("Decode failed: frame too short")
                    continue

                header_string = "".join(str(int(b)) for b in rx_bits[:HEADER_BITS])
                message_len = int(header_string, 2)

                if message_len <= 0 or message_len > MAX_BITS:
                    print(f"Decode failed: invalid message length {message_len}")
                    continue

                if message_len % 8 != 0:
                    print(f"Decode failed: message length {message_len} not byte-aligned")
                    continue

                    
                required_bits = HEADER_BITS + message_len
                
                if len(rx_bits) < required_bits:
                    print(f"Decode failed: need {required_bits} bits, got {len(rx_bits)}")
                    continue

                message_bits = rx_bits[HEADER_BITS:required_bits]
                
                rx_bytes = bits_to_bytes(message_bits)

                KEY = b"PatTEws1o7HD5TpT-9IowWCdhxXvOKFXsQJxoAWf_lQ="
                rx_bytes = decrypt_file(rx_bytes, KEY)
                rx_bytes = decompress_file(rx_bytes)

                if rx_bytes == last_message:
                    print("Skipping duplicate message")
                    continue

                last_message = rx_bytes
                window_messages.append(rx_bytes)
                print(f"Got message {len(window_messages)} in current window")
                print(f"Content: {rx_bytes[:200]}")

                plt.figure(figsize=(6, 6))
                plt.scatter(np.real(rx_symbols), np.imag(rx_symbols), color='red', s=1, label='Received QAM Symbols')
                plt.title('Data Symbols After Equalization')
                plt.xlabel('Real Component')
                plt.ylabel('Imaginary Component')
                plt.grid(True)
                plt.legend()
                plt.savefig("constellation.png", dpi=100)
                plt.close()

            except Exception:
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        if window_messages:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            output_path = OUTPUT_DIR / f"rx_{timestamp}.log"
            output_path.write_bytes(b"\n".join(window_messages))
            print(f"Saved {len(window_messages)} messages to {output_path}")
        print("\nReceiver stopped.")

if __name__ == "__main__":
    main()
