import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys
import argparse
from cryptography.fernet import Fernet

from cosmos import *
from digicomm import *
from helpers import *

import adi

# Directory for saving plots
dir_plots = 'plots/'

# ---------------------------------------------------------------
# Args.
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Encrypt + transmit a file over the Pluto SDR link.")
parser.add_argument(
    "file_path",
    nargs="?",
    default="/Users/ryanli/Desktop/Test.txt",
    help="Path of the file to encrypt and transmit (defaults to the test file).",
)
args = parser.parse_args()

# ---------------------------------------------------------------
# Setup.
# ---------------------------------------------------------------
sdr_tx = adi.Pluto("usb:1.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_channel(7)
tx.set_power_level(90)

# ---------------------------------------------------------------
# Generate symbols from file data.
# ---------------------------------------------------------------

M = 16
k = int(np.log2(M))
header_bits = 32
max_bits = 19968

KEY = b"PatTEws1o7HD5TpT-9IowWCdhxXvOKFXsQJxoAWf_lQ="


print(f"Reading file to transmit: {args.file_path}")
filebytes = file_to_bytes(args.file_path)
filebytes = compress_file(filebytes)
filebytes = encrypt_file(filebytes, KEY)


filebits = bytes_to_bits(filebytes)
if len(filebits) > max_bits:
    raise ValueError("file too big bud")

message_len = len(filebits)
binary_message_len = format(message_len, f'0{header_bits}b')
header_bit_list = [int(digit) for digit in binary_message_len]
header_bit_values = np.array(header_bit_list)
padding_zeros = max_bits - message_len
pad_bits = np.random.randint(0, 2, padding_zeros)
padded_filebits = np.concatenate((filebits, pad_bits))
all_bits = np.concatenate((header_bit_values, padded_filebits))

tx_symbols, remainder = bits_to_qam_symbols(all_bits, M)
constellation = get_qam_constellation(M, Es=1)
print(f"num_transmit_symbols = {len(tx_symbols)}, remainder = {remainder}")
# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)
time.sleep(50)
