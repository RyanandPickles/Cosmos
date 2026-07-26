"""
receive.py
----------
Receives jpg image frames over the Pluto SDR link -- the RX-side
counterpart to send.py. Uses:
    cosmos.PlutoReceiver           -- physical layer (sync, equalization, radio)
    digicomm.qam_symbols_to_bits   -- demodulate equalized symbols back to bits
    ldpc.LDPCCode                  -- FEC decode, built ONCE and reused every frame
    helpers.bits_to_jpg            -- write recovered bits out as a jpg file

*** THE "CONFIG" BLOCK BELOW MUST EXACTLY MATCH send.py ***
Same LDPC parameters, same QAM order, same FRAME_PAYLOAD_BYTES -- these
determine num_transmit_symbols, which PlutoReceiver needs to know ahead
of time in order to synchronize each frame.

WHY THIS DOESN'T JUST TRUST THE LDPC OUTPUT:
LDPC's bit-flipping decoder can fail silently -- see ldpc.py's docstring
and the reliability note in its __main__ block. A failed block doesn't
raise an error, it just returns its last (possibly still-wrong) guess.
So every frame here is protected by a CRC32 check on top of LDPC: if a
frame's checksum doesn't match after decoding, the whole frame is
dropped and the last known-good frame is kept, instead of writing
corrupted image data to disk. This is the "CRC-catches-failures" design
mentioned in ldpc.py's module docstring, actually wired up.

NOTE: this has NOT been tested against real Pluto hardware -- no SDR /
`adi` module is available in the environment this was written in. The
encode->modulate->demodulate->decode->CRC round trip has been verified
numerically (including under simulated noise), but the actual radio
calls (sample rate, rx_buffer_size, symbol counts) are unverified --
you'll likely need to tune rx_buffer_size upward if num_rx_symbols
(printed at startup) doesn't fit comfortably inside it once you account
for preamble + pulse-shaping length in samples.
"""

import argparse
import os
import struct
import time
import zlib

import numpy as np
import adi

from cosmos import PlutoReceiver
from digicomm import qam_symbols_to_bits
from helpers import bits_to_jpg
from ldpc import LDPCCode, uint8_to_bitstring

# ================= CONFIG -- MUST MATCH send.py EXACTLY =================
LDPC_K = 256
LDPC_M = 256
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42

QAM_ORDER = 4

FRAME_PAYLOAD_BYTES = 2048
HEADER_BYTES = 8
# ===========================================================================

LDPC_MAX_ITERATIONS = 15   # bump this before reaching for a stronger decoder --
                           # see the fps/error-correction tradeoff notes; more
                           # iterations costs time roughly linearly, so budget
                           # against your target fps.

SDR_RX_URI = "usb:1.11.5"
SAMPLE_RATE = 6_000_000
RX_BUFFER_SIZE = 500e3     # TUNE THIS -- must comfortably fit preamble +
                           # pulse-shaped data length in samples; verify
                           # against your actual num_rx_symbols (printed below)
CHANNEL = 7
RX_GAIN_LEVEL = 80


def compute_frame_sizing(code, qam_order, frame_payload_bytes):
    """Must produce the identical numbers send.py computes -- see that file."""
    frame_bits = frame_payload_bytes * 8
    num_blocks = -(-frame_bits // code.k)
    total_encoded_bits = num_blocks * code.n
    bits_per_symbol = int(np.log2(qam_order))
    num_qam_symbols = -(-total_encoded_bits // bits_per_symbol)
    return num_qam_symbols, total_encoded_bits


def unpack_frame(decoded_bit_string, payload_size):
    """
    Reverses send.py's pack_frame(): pulls the length header out, slices
    out exactly that many data bits, and validates the CRC32 footer.
    Returns (data_bits, ok) -- ok is False if the length field itself
    looks corrupted (out of range) or the CRC doesn't match. data_bits
    is only meaningful when ok is True.
    """
    data_capacity = payload_size - HEADER_BYTES

    length = int(decoded_bit_string[0:32], 2)
    if not (0 <= length <= data_capacity):
        # Length field itself got corrupted past recognition -- bail out
        # before trying to slice with a nonsense length.
        return None, False

    data_bits = decoded_bit_string[32:32 + length * 8]
    crc_bits = decoded_bit_string[32 + length * 8: 32 + length * 8 + 32]
    if len(crc_bits) != 32:
        return None, False

    crc_received = int(crc_bits, 2)
    data_bytes = bytes(int(data_bits[i:i + 8], 2) for i in range(0, len(data_bits), 8))
    crc_computed = zlib.crc32(data_bytes) & 0xFFFFFFFF

    return data_bits, (crc_computed == crc_received)


def main():
    parser = argparse.ArgumentParser(description="Receive jpg frames over the Pluto SDR link.")
    parser.add_argument('--out', default='received_frames',
                         help="directory to write verified-good frames into")
    parser.add_argument('--latest-name', default='latest.jpg',
                         help="filename (inside --out) that always holds the most recent good frame")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Build the LDPC code ONCE -- must match send.py's k/m/seed exactly or
    # every frame will fail to decode (different code entirely).
    code = LDPCCode(k=LDPC_K, m=LDPC_M, col_weight=LDPC_COL_WEIGHT, seed=LDPC_SEED)
    num_rx_symbols, total_encoded_bits = compute_frame_sizing(code, QAM_ORDER, FRAME_PAYLOAD_BYTES)
    original_length = FRAME_PAYLOAD_BYTES * 8
    print(f"LDPC rate: {code.k}/{code.n} ({code.k/code.n:.2f})  "
          f"Expecting {num_rx_symbols} QAM symbols/frame")

    # Radio setup -- once, not per frame.
    sdr_rx = adi.Pluto(SDR_RX_URI)
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_sample_rate(SAMPLE_RATE)
    rx.set_buffer_size(RX_BUFFER_SIZE)
    rx.set_channel(CHANNEL)
    rx.set_gain_level(RX_GAIN_LEVEL)

    # Our QAM symbols are genuinely complex-valued (not real PAM duplicated
    # onto I/Q), so tell the receiver not to force them real.
    rx.desired_transmit_symbols_real = False
    rx.num_transmit_symbols = num_rx_symbols

    frame_idx = 0
    frames_ok = 0
    frames_dropped = 0

    while True:
        t_start = time.time()
        frame_idx += 1

        try:
            rx_symbols = rx.receive()
        except Exception as e:
            frames_dropped += 1
            print(f"[frame {frame_idx}] receive/sync failed ({e}) -- dropped, kept last good frame")
            continue

        # --- demodulate ---
        raw_bits = qam_symbols_to_bits(rx_symbols, QAM_ORDER)
        # drop the QAM/symbol-count alignment padding tail, keeping only
        # the actual LDPC codeword bits before reshaping into blocks
        raw_bits = raw_bits[:total_encoded_bits]
        encoded_bit_string = uint8_to_bitstring(raw_bits)

        # --- LDPC decode ---
        decoded_bit_string, num_blocks_failed = code.decode(
            encoded_bit_string, original_length, max_iterations=LDPC_MAX_ITERATIONS
        )

        # --- CRC check (catches LDPC's silent-failure blocks) ---
        data_bits, ok = unpack_frame(decoded_bit_string, FRAME_PAYLOAD_BYTES)

        elapsed = time.time() - t_start

        if ok:
            frames_ok += 1
            data_bytes = bytes(int(data_bits[i:i + 8], 2) for i in range(0, len(data_bits), 8))
            out_path = os.path.join(args.out, f"frame_{frame_idx:06d}.jpg")
            with open(out_path, 'wb') as f:
                f.write(data_bytes)
            latest_path = os.path.join(args.out, args.latest_name)
            with open(latest_path, 'wb') as f:
                f.write(data_bytes)
            status = "OK"
        else:
            frames_dropped += 1
            status = "DROPPED (CRC/length invalid -- kept last good frame)"

        print(f"[frame {frame_idx}] {status} | LDPC blocks failed: {num_blocks_failed} "
              f"| {elapsed*1000:.1f} ms | ok={frames_ok} dropped={frames_dropped}")


if __name__ == "__main__":
    main()
