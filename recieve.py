#used to parse command line arguments, terminal strings --> python variables
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

LDPC_K = 256
LDPC_M = 256
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42

QAM_ORDER = 16

FRAME_PAYLOAD_BYTES = 2048
HEADER_BYTES = 8

LDPC_MAX_ITERATIONS = 15  

SDR_RX_URI = "usb:1.11.5"
SAMPLE_RATE = 6_000_000
RX_BUFFER_SIZE = 230e3    #tune
CHANNEL = 7
RX_GAIN_LEVEL = 80


def compute_frame_sizing(code, qam_order, frame_payload_bytes):
    """
Description:
    Produces numbers send.py computes
    """
    frame_bits = frame_payload_bytes * 8
    number_blocks = -(-frame_bits // code.k)
    total_encoded_bits = number_blocks * code.n
    bits_per_symbol = int(np.log2(qam_order))
    number_qam_symbols = -(-total_encoded_bits // bits_per_symbol)
    return number_qam_symbols, total_encoded_bits


def unpack_frame(decoded_bit_string, payload_size):
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

    return data_bits, (crc_computed == crc_received)


def main():
    parser = argparse.ArgumentParser(description="recieve jpg frames")
    parser.add_argument('--out', default='received_frames')
    parser.add_argument('--latest-name', default='latest.jpg')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    code = LDPCCode(k=LDPC_K, m=LDPC_M, column_weight=LDPC_COL_WEIGHT, seed=LDPC_SEED)
    number_rx_symbols, total_encoded_bits = compute_frame_sizing(code, QAM_ORDER, FRAME_PAYLOAD_BYTES)
    original_length = FRAME_PAYLOAD_BYTES * 8
    print(f"LDPC rate: {code.k}/{code.n} ({code.k/code.n:.2f})  "
          f"Expecting {number_rx_symbols} QAM symbols/frame")

    sdr_rx = adi.Pluto(SDR_RX_URI)
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_sample_rate(SAMPLE_RATE)
    rx.set_buffer_size(RX_BUFFER_SIZE)
    rx.set_channel(CHANNEL)
    rx.set_gain_level(RX_GAIN_LEVEL)

    rx.desired_transmit_symbols_real = False
    rx.num_transmit_symbols = number_rx_symbols

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

        raw_bits = qam_symbols_to_bits(rx_symbols, QAM_ORDER)
        raw_bits = raw_bits[:total_encoded_bits]
        encoded_bit_string = uint8_to_bitstring(raw_bits)

        decoded_bit_string, num_blocks_failed = code.decode(
            encoded_bit_string, original_length, max_iterations=LDPC_MAX_ITERATIONS
        )

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
