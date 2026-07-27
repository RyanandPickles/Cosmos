"""
send.py
-------
Streams jpg image frames over the Pluto SDR link, using:
    helpers.jpg_to_bits        -- read a frame off disk as bits
    ldpc.LDPCCode              -- FEC, built ONCE and reused every frame
    digicomm.bits_to_qam_symbols -- modulate encoded bits onto QAM symbols
    cosmos.PlutoTransmitter    -- physical layer (preamble, pulse shaping, radio)

FRAME FORMAT (fixed size, so the receiver always knows exactly how many
QAM symbols to expect -- this is what lets main_rx.py-style code avoid
needing an explicit "frame length" negotiation over the air):

    [4 bytes] payload length (uint32, big-endian)
    [N bytes] jpg payload  (N = payload length)
    [4 bytes] CRC32 of the jpg payload (uint32, big-endian)
    [pad]     zero bytes out to FRAME_PAYLOAD_BYTES total

This whole fixed-size buffer is what gets LDPC-encoded and QAM-modulated,
every frame, giving a fixed number of transmitted symbols per frame.

*** EVERYTHING IN THE "CONFIG" BLOCK BELOW MUST EXACTLY MATCH receive.py ***
Both sides derive the number of QAM symbols/frame from these constants --
if they don't match, frame_synchronization on the RX side will be looking
for the wrong buffer length and every frame will fail to sync.

NOTE: this has NOT been tested against real Pluto hardware -- there's no
SDR / `adi` module available in the environment this was written in. The
encode -> modulate -> demodulate -> decode -> CRC-check round trip has
been verified numerically (including under simulated noise), but the
actual radio calls (sample rate, buffer sizing, timing) are unverified
and you should confirm them against your link before relying on this.
"""

import argparse
import glob
import struct
import time
import zlib

import numpy as np
import adi

from cosmos import PlutoTransmitter
from digicomm import bits_to_qam_symbols
from helpers import jpg_to_bits
from ldpc import LDPCCode, bitstring_to_uint8

# ================= CONFIG -- MUST MATCH receive.py EXACTLY =================
LDPC_K = 256              # message bits per LDPC block
LDPC_M = 256              # parity bits per LDPC block (k=m -> rate 1/2)
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42            # TX and RX must build the identical random code

QAM_ORDER = 16             # 4 = QPSK. Bump to 16 for more throughput, at the
                          # cost of needing a cleaner channel to hit the
                          # same block-success rate (more bits/symbol =
                          # closer constellation points = less noise margin).

FRAME_PAYLOAD_BYTES = 2048  # fixed size of every frame's raw payload buffer
HEADER_BYTES = 8            # 4 bytes length + 4 bytes CRC32
# =============================================================================

SDR_TX_URI = "usb:1.12.5"
SAMPLE_RATE = 6_000_000
CHANNEL = 7
TX_POWER_LEVEL = 95


def compute_num_tx_symbols(code, qam_order, frame_payload_bytes):
    """
    Deterministically computes how many QAM symbols one fixed-size frame
    turns into, from the LDPC code + QAM order alone -- no data-dependent
    length needs to cross the air, both sides just compute this the same way.
    """
    frame_bits = frame_payload_bytes * 8
    num_blocks = -(-frame_bits // code.k)          # ceil(frame_bits / k)
    total_encoded_bits = num_blocks * code.n
    bits_per_symbol = int(np.log2(qam_order))
    num_qam_symbols = -(-total_encoded_bits // bits_per_symbol)  # ceil
    return num_qam_symbols, total_encoded_bits


def pack_frame(jpg_bytes, payload_size):
    """
    Wraps raw jpg bytes in [length][data][crc32][zero-pad] out to a fixed
    payload_size. Raises if the jpg doesn't fit -- silently truncating
    image data would corrupt it, so this fails loudly instead.
    """
    data_capacity = payload_size - HEADER_BYTES
    if len(jpg_bytes) > data_capacity:
        raise ValueError(
            f"Frame is {len(jpg_bytes)} bytes but only {data_capacity} bytes "
            f"fit in a FRAME_PAYLOAD_BYTES={payload_size} budget. "
            f"Compress/resize the image more, or raise FRAME_PAYLOAD_BYTES "
            f"(and update it identically in receive.py)."
        )
    crc = zlib.crc32(jpg_bytes) & 0xFFFFFFFF
    header = struct.pack('>I', len(jpg_bytes))
    footer = struct.pack('>I', crc)
    pad = bytes(data_capacity - len(jpg_bytes))
    return header + jpg_bytes + footer + pad


def bytes_to_bitstring(data):
    return ''.join(format(b, '08b') for b in data)


def main():
    parser = argparse.ArgumentParser(description="Stream jpg frames over the Pluto SDR link.")
    parser.add_argument('--frames', default='frames/*.jpg',
                         help="glob pattern for jpg frames to send, looped in sorted order")
    parser.add_argument('--fps', type=float, default=10.0)
    args = parser.parse_args()

    # Build the LDPC code ONCE. This is the entire point of LDPCCode --
    # rebuilding the sparse H every frame would blow the frame time budget.
    code = LDPCCode(k=LDPC_K, m=LDPC_M, column_weight=LDPC_COL_WEIGHT, seed=LDPC_SEED)
    num_tx_symbols, total_encoded_bits = compute_num_tx_symbols(code, QAM_ORDER, FRAME_PAYLOAD_BYTES)
    print(f"LDPC rate: {code.k}/{code.n} ({code.k/code.n:.2f})  "
          f"Frame payload: {FRAME_PAYLOAD_BYTES} bytes -> {num_tx_symbols} QAM symbols/frame")

    # Radio setup -- once, not per frame.
    sdr_tx = adi.Pluto(SDR_TX_URI)
    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_sample_rate(SAMPLE_RATE)
    tx.set_channel(CHANNEL)
    tx.set_power_level(TX_POWER_LEVEL)

    frame_paths = sorted(glob.glob(args.frames))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames matched glob '{args.frames}'. Point --frames at your "
            f"captured/compressed jpg frames (this script doesn't do camera "
            f"capture itself, it just streams whatever jpgs it's given)."
        )
    print(f"Streaming {len(frame_paths)} frame(s) from '{args.frames}' at target {args.fps} fps")

    frame_interval = 1.0 / args.fps
    frame_idx = 0

    while True:
        t_start = time.time()

        path = frame_paths[frame_idx % len(frame_paths)]
        frame_idx += 1

        # --- read + pack ---
        with open(path, 'rb') as f:
            jpg_bytes = f.read()
        packed = pack_frame(jpg_bytes, FRAME_PAYLOAD_BYTES)
        bit_string = bytes_to_bitstring(packed)

        # --- LDPC encode ---
        encoded_bits, original_length = code.encode(bit_string)

        # --- QAM modulate ---
        bits_array = bitstring_to_uint8(encoded_bits)
        qam_symbols, _ = bits_to_qam_symbols(bits_array, QAM_ORDER)

        # Safety net: should already be exactly num_tx_symbols long given
        # the deterministic sizing above, but pad/truncate defensively in
        # case the constants above ever get out of sync with each other.
        if len(qam_symbols) < num_tx_symbols:
            qam_symbols = np.concatenate([
                qam_symbols,
                np.zeros(num_tx_symbols - len(qam_symbols), dtype=qam_symbols.dtype),
            ])
        elif len(qam_symbols) > num_tx_symbols:
            qam_symbols = qam_symbols[:num_tx_symbols]

        # --- transmit ---
        tx.transmit(qam_symbols)

        elapsed = time.time() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"[frame {frame_idx}] encode+transmit took {elapsed*1000:.1f} ms "
                  f"-- can't hit {args.fps} fps at this LDPC_K/QAM_ORDER/FRAME_PAYLOAD_BYTES, "
                  f"running flat-out instead")


if __name__ == "__main__":
    main()
