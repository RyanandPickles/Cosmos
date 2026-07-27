"""
recieve.py
----------
Receives an H.264 elementary stream over the Pluto SDR link -- the
RX-side counterpart to send.py. Reassembles the continuous stream from
received chunks and writes it out for a real H.264 decoder/player to
consume downstream.

WHY A FAILED CHUNK IS SKIPPED RATHER THAN "KEEP THE LAST GOOD FRAME":
A corrupted chunk is a hole in the MIDDLE of a continuous bitstream --
there's no sensible "last good chunk" to redisplay, since the
reassembled stream is consumed continuously by a decoder, not
frame-by-frame the way standalone images would be.

Instead: a chunk that fails its CRC check is SKIPPED (not written),
leaving a real gap in the reassembled output, logged with its chunk
index and byte offset so it's debuggable. H.264 NAL units use
self-synchronizing start codes (0x000001), so a real H.264 decoder
downstream can typically resync at the next intact NAL boundary after
a gap -- but everything between the gap and that next boundary is
lost. This is expected, standard behavior for lossy H.264 delivery,
not a bug in this reassembly logic.
"""

import argparse
import os
import time
import zlib

import numpy as np
import adi

from cosmos import PlutoReceiver
from digicomm import qam_symbols_to_bits
from ldpc import LDPCCode, uint8_to_bitstring

# ================= CONFIG -- MUST MATCH send.py EXACTLY =================
LDPC_K = 256
LDPC_M = 256
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42
QAM_ORDER = 16
FRAME_PAYLOAD_BYTES = 2048
HEADER_BYTES = 8
# =================================================================================

LDPC_MAX_ITERATIONS = 15

SDR_RX_URI = "usb:0.1.5"
SAMPLE_RATE = 6_000_000
RX_BUFFER_SIZE = 500e3
CHANNEL = 7
RX_GAIN_LEVEL = 95

DATA_CAPACITY = FRAME_PAYLOAD_BYTES - HEADER_BYTES


def compute_frame_sizing(code, qam_order, frame_payload_bytes):
    frame_bits = frame_payload_bytes * 8
    num_blocks = -(-frame_bits // code.k)
    total_encoded_bits = num_blocks * code.n
    bits_per_symbol = int(np.log2(qam_order))
    num_qam_symbols = -(-total_encoded_bits // bits_per_symbol)
    return num_qam_symbols, total_encoded_bits


def unpack_frame(decoded_bit_string, payload_size):
    """UNCHANGED from recieve.py -- reads [length][data][crc32], validates."""
    data_capacity = payload_size - HEADER_BYTES
    length = int(decoded_bit_string[0:32], 2)
    if not (0 <= length <= data_capacity):
        return None, False

    data_bits = decoded_bit_string[32:32 + length * 8]
    crc_bits = decoded_bit_string[32 + length * 8: 32 + length * 8 + 32]
    if len(crc_bits) != 32:
        return None, False

    crc_received = int(crc_bits, 2)
    data_bytes = bytes(int(data_bits[i:i + 8], 2) for i in range(0, len(data_bits), 8))
    crc_computed = zlib.crc32(data_bytes) & 0xFFFFFFFF

    return data_bytes, (crc_computed == crc_received)


def main():
    parser = argparse.ArgumentParser(description="Receive a continuous H.264 elementary stream over the SDR link.")
    parser.add_argument('--out', default='received_stream.h264',
                         help="path to write the reassembled H.264 elementary stream to (feed this to a decoder/player)")
    args = parser.parse_args()

    code = LDPCCode(k=LDPC_K, m=LDPC_M, column_weight=LDPC_COL_WEIGHT, seed=LDPC_SEED)
    num_rx_symbols, total_encoded_bits = compute_frame_sizing(code, QAM_ORDER, FRAME_PAYLOAD_BYTES)
    original_length = FRAME_PAYLOAD_BYTES * 8
    print(f"LDPC rate: {code.k}/{code.n} ({code.k/code.n:.2f})  "
          f"Expecting {num_rx_symbols} QAM symbols/chunk")

    sdr_rx = adi.Pluto(SDR_RX_URI)
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_sample_rate(SAMPLE_RATE)
    rx.set_buffer_size(RX_BUFFER_SIZE)
    rx.set_channel(CHANNEL)
    rx.set_gain_level(RX_GAIN_LEVEL)

    rx.desired_transmit_symbols_real = False
    rx.num_transmit_symbols = num_rx_symbols

    chunk_idx = 0
    chunks_ok = 0
    chunks_dropped = 0
    byte_offset = 0  # running position in the reassembled stream, for gap logging

    out_file = open(args.out, 'wb')
    print(f"Writing reassembled stream to '{args.out}' (append a real H.264 decoder/player downstream to consume it)")

    while True:
        t_start = time.time()
        chunk_idx += 1

        try:
            t0 = time.time()
            rx_symbols = rx.receive()
            print(f"  rx.receive() alone took {(time.time()-t0)*1000:.1f} ms")
        except Exception as e:
            chunks_dropped += 1
            print(f"[chunk {chunk_idx}] receive/sync failed ({e}) -- GAP at byte offset {byte_offset}")
            continue

        raw_bits = qam_symbols_to_bits(rx_symbols, QAM_ORDER)
        raw_bits = raw_bits[:total_encoded_bits]
        encoded_bit_string = uint8_to_bitstring(raw_bits)

        decoded_bit_string, num_blocks_failed = code.decode(
            encoded_bit_string, original_length, max_iterations=LDPC_MAX_ITERATIONS
        )

        data_bytes, ok = unpack_frame(decoded_bit_string, FRAME_PAYLOAD_BYTES)

        elapsed = time.time() - t_start

        if ok:
            chunks_ok += 1
            out_file.write(data_bytes)
            out_file.flush()  # so a downstream decoder tailing this file sees data promptly
            byte_offset += len(data_bytes)
            status = "OK"
        else:
            chunks_dropped += 1
            status = f"GAP at byte offset {byte_offset} (chunk skipped, not written)"

        print(f"[chunk {chunk_idx}] {status} | LDPC blocks failed: {num_blocks_failed} "
              f"| {elapsed*1000:.1f} ms | ok={chunks_ok} dropped={chunks_dropped}")


if __name__ == "__main__":
    main()
