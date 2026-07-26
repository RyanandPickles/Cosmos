"""
send.py
-------
Master TX file. Run this on the computer connected to the TRANSMIT
ADALM-Pluto. Captures webcam frames, compresses each one (compression.py),
LDPC-encodes + QAM-modulates it (transport.py), and continuously
transmits it (cosmos.py / PlutoTransmitter).

*** IMPORTANT: LINK_CONFIG below must exactly match receive.py's ***
*** LINK_CONFIG on the other computer (same M, ldpc_k, ldpc_m,   ***
*** ldpc_seed, num_transmit_symbols). Same for CAM_WIDTH/HEIGHT   ***
*** and SAMPLE_RATE if you change them.                           ***

Usage:
    python3 send.py
    (Ctrl+C to stop cleanly.)
"""

import time
import cv2
import numpy as np

from cosmos import PlutoTransmitter
import compression
import transport

import adi

# =================================================================
# Configuration -- must match receive.py
# =================================================================
PLUTO_TX_URI = "usb:1.12.5"     # from main_tx.py
CHANNEL = 7                     # from main_tx.py
POWER_LEVEL = 95                # from main_tx.py
SAMPLE_RATE = 5e6               # samples/sec. Bumped up from the 1e6 demo
                                 # default -- throughput scales directly
                                 # with this (see fps estimate printed below).
                                 # Pluto's practical ceiling over USB with
                                 # pyadi-iio is well under its 61e6 max;
                                 # if you see dropped/garbled buffers, lower
                                 # this first.

CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 160, 120   # transmit resolution. Small on purpose --
                                    # this link has to fit a whole compressed
                                    # frame into a few thousand QAM symbols.

# LDPC + QAM parameters -- must match receive.py's LINK_CONFIG exactly,
# since TX and RX independently build the SAME parity-check matrices from
# these (see ldpc.generate_ldpc_matrices' seed argument).
NUM_TRANSMIT_SYMBOLS = 20000
QAM_ORDER = 16          # 16-QAM = 4 bits/symbol. Try 4 (QPSK, more robust,
                         # less throughput) or 64 (more throughput, needs a
                         # cleaner channel) depending on your link quality.
LDPC_K = 128
LDPC_M = 128             # rate = K/(K+M) = 0.5. Lower M = higher rate/more
                          # payload throughput but weaker error correction.
LDPC_SEED = 42            # MUST match receive.py

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
    print(f"Estimated max frame rate at {SAMPLE_RATE/1e6:.1f} Msps: {est_fps:.1f} fps "
          f"(radio-limited; actual fps will also depend on webcam/compression/USB speed)")

    # ---------------------------------------------------------------
    # Radio setup
    # ---------------------------------------------------------------
    sdr_tx = adi.Pluto(PLUTO_TX_URI)
    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_channel(CHANNEL)
    tx.set_power_level(POWER_LEVEL)
    tx.set_sample_rate(int(SAMPLE_RATE))

    # ---------------------------------------------------------------
    # Camera + compressor setup
    # ---------------------------------------------------------------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}.")

    compressor = compression.AdaptiveFrameCompressor(
        max_bytes=LINK_CONFIG.max_payload_bytes,
        width=CAM_WIDTH, height=CAM_HEIGHT,
    )

    print("Transmitting... (Ctrl+C to stop)")
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: camera read failed, skipping frame.")
                continue

            jpg, quality, size_used = compressor.compress(frame)
            symbols = transport.encode_for_tx(jpg, LINK_CONFIG)

            # tx.transmit() uses a cyclic buffer (set in PlutoTransmitter.set_sdr),
            # so this frame repeats continuously over the air until the next
            # call replaces it -- that's what gives the receiver a continuous
            # signal to lock onto even if this loop briefly stalls.
            tx.transmit(symbols)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                print(f"Frame {frame_count} | {len(jpg)}B @ q{quality} | "
                      f"{frame_count/elapsed:.1f} fps avg | size={size_used}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tx.stop_transmission()
        cap.release()


if __name__ == "__main__":
    main()
