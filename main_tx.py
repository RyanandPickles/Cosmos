import numpy as np
import time
from datetime import datetime

from cosmos import PlutoTransmitter
from digicomm import bits_to_qam_symbols
from helpers import file_to_bytes, bytes_to_bits, compress_file, encrypt_file

M = 16
HEADER_BITS = 32
MAX_BITS = 7968
KEY = b"PatTEws1o7HD5TpT-9IowWCdhxXvOKFXsQJxoAWf_lQ="


def transmit_file(file_path, tx, sleep_seconds=1):
    print(f"Transmitting: {file_path}")
    filebytes = file_to_bytes(file_path)
    filebytes = compress_file(filebytes)
    filebytes = encrypt_file(filebytes, KEY)

    filebits = bytes_to_bits(filebytes)
    if len(filebits) > MAX_BITS:
        raise ValueError("file too big bud")

    message_len = len(filebits)
    binary_message_len = format(message_len, f'0{HEADER_BITS}b')
    header_bit_values = np.array([int(d) for d in binary_message_len])
    pad_bits = np.random.randint(0, 2, MAX_BITS - message_len)
    all_bits = np.concatenate((header_bit_values, filebits, pad_bits))

    tx_symbols, remainder = bits_to_qam_symbols(all_bits, M)
    print(f"num_transmit_symbols = {len(tx_symbols)}, remainder = {remainder}")

    tx.stop_transmission()
    tx.transmit(tx_symbols)
    time.sleep(sleep_seconds)
    print("Done transmitting.")


if __name__ == "__main__":
    import argparse
    import adi

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", default="/Users/ryanli/Desktop/Test.txt")
    parser.add_argument("--uri", default="usb:1.1.5")
    args = parser.parse_args()

    sdr_tx = adi.Pluto(args.uri)
    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_channel(7)
    tx.set_power_level(70)

    transmit_file(args.file_path, tx)