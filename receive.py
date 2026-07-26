"""
receive.py
----------
Master RX file. Run this on the computer connected to the RECEIVE
ADALM-Pluto. Continuously receives QAM symbols (cosmos.py /
PlutoReceiver), LDPC-decodes + demodulates + integrity-checks each frame
(transport.py), decompresses the good ones (compression.py), and
displays the video.

*** IMPORTANT: LINK_CONFIG below must exactly match send.py's LINK_CONFIG ***
*** on the other computer (same M, ldpc_k, ldpc_m, ldpc_seed,             ***
*** num_transmit_symbols). Same for SAMPLE_RATE if you change it.        ***

Usage:
    python3 receive.py
    (press 'q' in the video window, or Ctrl+C in the terminal, to stop.)
"""

import time
import cv2
import numpy as np

from cosmos import PlutoReceiver
import compression
import transport

import adi

# =================================================================
# Configuration -- must match send.py
# =================================================================
PLUTO_RX_URI = "usb:1.11.5"    # from main_rx.py
CHANNEL = 7                    # from main_rx.py
GAIN_LEVEL = 80                # from main_rx.py
SAMPLE_RATE = 8e6              # MUST match send.py's SAMPLE_RATE

NUM_TRANSMIT_SYMBOLS = 20000
QAM_ORDER = 16
LDPC_K = 128
LDPC_M = 128
LDPC_SEED = 42                 # MUST match send.py

LINK_CONFIG = transport.LinkConfig(
    num_transmit_symbols=NUM_TRANSMIT_SYMBOLS,
    M=QAM_ORDER,
    ldpc_k=LDPC_K,
    ldpc_m=LDPC_M,
    ldpc_seed=LDPC_SEED,
)


def main():
    print(LINK_CONFIG)
    est_fps = transport.max_achievable_fps(LINK_CONFIG, SAMPLE_RATE)
    print(f"Estimated max frame rate at {SAMPLE_RATE/1e6:.1f} Msps: {est_fps:.1f} fps")

    # ---------------------------------------------------------------
    # Radio setup
    # ---------------------------------------------------------------
    sdr_rx = adi.Pluto(PLUTO_RX_URI)
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_sample_rate(int(SAMPLE_RATE))
    rx.set_buffer_size(transport.recommended_rx_buffer_size(LINK_CONFIG))
    rx.set_channel(CHANNEL)
    rx.set_gain_level(GAIN_LEVEL)

    # Our payload is complex QAM data, not the real-PAM demo signal --
    # tell PlutoReceiver not to force it back to real-valued.
    rx.desired_transmit_symbols_real = False
    rx.num_transmit_symbols = LINK_CONFIG.num_transmit_symbols

    print("Receiving... (press 'q' in the video window, or Ctrl+C, to stop)")
    window_name = "Cosmos RX"
    last_good_frame = None
    frame_count = 0
    good_count = 0
    t_start = time.time()

    try:
        while True:
            try:
                symbols = rx.receive()
            except Exception as e:
                # Frame/timing sync can occasionally fail to find a valid
                # preamble in a given buffer capture -- just try the next one
                # rather than crashing the whole stream.
                print(f"Warning: receive() failed ({e}); retrying.")
                continue

            jpg_bytes, info = transport.decode_from_rx(symbols, LINK_CONFIG)
            frame_count += 1

            if jpg_bytes is not None:
                frame = compression.decode_frame(jpg_bytes)
                if frame is not None:
                    last_good_frame = frame
                    good_count += 1

            display_frame = last_good_frame
            if display_frame is not None:
                cv2.imshow(window_name, display_frame)

            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                print(f"Frame {frame_count} | good={good_count} "
                      f"({100*good_count/frame_count:.0f}%) | "
                      f"{frame_count/elapsed:.1f} fps avg | last info={info}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
