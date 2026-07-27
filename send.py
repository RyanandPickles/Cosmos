"""
send.py
-------
Streams a continuous H.264 elementary stream over the Pluto SDR link.

HOW FRAMING WORKS:
H.264 frames vary wildly in size (I-frames much bigger than P-frames),
so this doesn't send "one frame per transmission." Instead it treats
the H.264 output as one continuous byte stream and slices it into
fixed-size chunks, ignoring where video-frame boundaries fall. This is
the same approach real-world H.264-over-RTP uses (FU-A fragmentation)
-- H.264 NAL units have self-synchronizing start codes (0x000001), so
a decoder downstream can resync at the next NAL boundary even after a
gap.

Each chunk is wrapped as [length][data][crc32][zero-padding] via
pack_frame() before LDPC encoding, then QAM-modulated and transmitted
-- same as any other framed payload; pack_frame() doesn't care what
the bytes mean, only how many there are.

SOURCE can be:
  - A regular file path: reads the whole file once. Pass --loop to
    repeat it forever (useful for testing without a live encoder).
  - A named pipe (FIFO) path: reads block naturally as your friend's
    encoder writes new bytes, so this can run live against a real
    encoder process without needing the whole stream to exist upfront.
    Create one with: mkfifo my_stream.h264
    then point your friend's encoder's output at that same path.

NOTE ON WHAT --fps NOW MEANS: this paces CHUNK transmissions, not
video frames. How many video frames-per-second that translates to
depends on how many bytes your friend's encoder produces per video
frame relative to the chunk size (data_capacity bytes/chunk here).
"""

import argparse
import os
import struct
import time
import zlib

import numpy as np
import adi

from cosmos import PlutoTransmitter
from digicomm import bits_to_qam_symbols
from ldpc import LDPCCode, bitstring_to_uint8

# ================= CONFIG -- MUST MATCH recieve.py EXACTLY =================
LDPC_K = 256
LDPC_M = 256
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42
QAM_ORDER = 16
FRAME_PAYLOAD_BYTES = 2048
HEADER_BYTES = 8
# ====================================================================================

SDR_TX_URI = "usb:1.1.5"
SAMPLE_RATE = 6_000_000
CHANNEL = 7
TX_POWER_LEVEL = 95

DATA_CAPACITY = FRAME_PAYLOAD_BYTES - HEADER_BYTES  # bytes of real data per chunk


def compute_num_tx_symbols(code, qam_order, frame_payload_bytes):
    frame_bits = frame_payload_bytes * 8
    num_blocks = -(-frame_bits // code.k)
    total_encoded_bits = num_blocks * code.n
    bits_per_symbol = int(np.log2(qam_order))
    num_qam_symbols = -(-total_encoded_bits // bits_per_symbol)
    return num_qam_symbols, total_encoded_bits


def pack_frame(payload_bytes, payload_size):
    """UNCHANGED from send.py -- [length][data][crc32][zero-padding]."""
    data_capacity = payload_size - HEADER_BYTES
    if len(payload_bytes) > data_capacity:
        raise ValueError(f"chunk size {len(payload_bytes)} exceeds {data_capacity} bytes")
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    header = struct.pack('>I', len(payload_bytes))
    footer = struct.pack('>I', crc)
    pad = bytes(data_capacity - len(payload_bytes))
    return header + payload_bytes + footer + pad


def bytes_to_bitstring(data):
    return ''.join(format(b, '08b') for b in data)


def read_next_chunk(f, chunk_size):
    """
    Reads up to chunk_size bytes. For a regular file this returns fewer
    bytes (or b'') at EOF. For a named pipe/FIFO this blocks until data
    is available or the writer closes the pipe.
    """
    return f.read(chunk_size)


def main():
    parser = argparse.ArgumentParser(description="Stream a continuous H.264 elementary stream over the SDR link.")
    parser.add_argument('--source', required=True,
                         help="path to an H.264 elementary stream file, or a named pipe (FIFO) fed by your friend's encoder")
    parser.add_argument('--fps', type=float, default=10.0,
                         help="chunks per second to transmit (NOT video frames per second -- see module docstring)")
    parser.add_argument('--loop', action='store_true',
                         help="when --source is a regular file, restart from the beginning at EOF (ignored for a FIFO)")
    args = parser.parse_args()

    code = LDPCCode(k=LDPC_K, m=LDPC_M, column_weight=LDPC_COL_WEIGHT, seed=LDPC_SEED)
    num_tx_symbols, total_encoded_bits = compute_num_tx_symbols(code, QAM_ORDER, FRAME_PAYLOAD_BYTES)
    print(f"LDPC rate: {code.k}/{code.n} ({code.k/code.n:.2f})  "
          f"Chunk size: {DATA_CAPACITY} bytes -> {num_tx_symbols} QAM symbols/chunk")

    sdr_tx = adi.Pluto(SDR_TX_URI)
    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_sample_rate(SAMPLE_RATE)
    tx.set_channel(CHANNEL)
    tx.set_power_level(TX_POWER_LEVEL)

    is_fifo = os.path.exists(args.source) and os.stat(args.source).st_mode & 0o170000 == 0o010000
    print(f"Source: {args.source} ({'named pipe -- live' if is_fifo else 'regular file'})")

    frame_interval = 1.0 / args.fps
    chunk_index = 0

    f = open(args.source, 'rb')
    total_bytes_sent = 0

    while True:
        t_start = time.time()

        chunk = read_next_chunk(f, DATA_CAPACITY)

        if len(chunk) == 0:
            # EOF on a regular file (a FIFO would have blocked instead of returning empty,
            # unless the writer closed it, which counts as a real end too)
            if args.loop and not is_fifo:
                f.close()
                f = open(args.source, 'rb')
                print(f"[chunk {chunk_index}] reached end of source, looping back to start")
                continue
            else:
                print(f"[chunk {chunk_index}] end of stream, sent {total_bytes_sent} bytes total. Stopping.")
                break

        packed = pack_frame(chunk, FRAME_PAYLOAD_BYTES)
        bitstring = bytes_to_bitstring(packed)

        encoded_bits, original_length = code.encode(bitstring)
        bits_array = bitstring_to_uint8(encoded_bits)
        qam_symbols, _ = bits_to_qam_symbols(bits_array, QAM_ORDER)

        if len(qam_symbols) < num_tx_symbols:
            qam_symbols = np.concatenate([qam_symbols, np.zeros(num_tx_symbols - len(qam_symbols), dtype=qam_symbols.dtype)])
        elif len(qam_symbols) > num_tx_symbols:
            qam_symbols = qam_symbols[:num_tx_symbols]

        t0 = time.time()
        tx.transmit(qam_symbols)
        print(f"  tx.transmit() alone took {(time.time()-t0)*1000:.1f} ms")

        total_bytes_sent += len(chunk)
        chunk_index += 1

        elapsed = time.time() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"[chunk {chunk_index}] encode+transmit took {elapsed*1000:.1f} ms -- can't hit {args.fps} chunks/sec")

    f.close()


if __name__ == "__main__":
    main()
