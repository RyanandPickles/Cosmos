import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys

from cosmos import *
from digicomm import *
from helpers import file_to_bytes, bytes_to_bits

import adi

# Directory for saving plots
dir_plots = 'plots/'

# ---------------------------------------------------------------
# Setup.
# --------------------------------------3-------------------------
sdr_tx = adi.Pluto("usb:0.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_channel(7)
tx.set_power_level(95)

# ---------------------------------------------------------------
# Generate random symbols.
# ---------------------------------------------------------------

M=16
k= int(np.log2(M))
header_bits=32
max_bits=800
filebytes = file_to_bytes("/Users/ryanli/Desktop/Test.txt")
filebits = bytes_to_bits(filebytes)
if len(filebits) > max_bits:
    raise ValueError("file too big bud")

message_len = len(filebits)
binary_message_len=format(message_len, f'0{header_bits}b')
header_bit_list = [int(digit) for digit in binary_message_len]
header_bit_values = np.array(header_bit_list)
padding_zeros = max_bits - message_len
padded_filebits = np.pad(filebits, (0, padding_zeros))
all_bits = np.concatenate((header_bit_values, padded_filebits))

tx_symbols, remainder = bits_to_qam_symbols(all_bits, M)
constellation = get_qam_constellation(M, Es=1)
print(f"num_transmit_symbols = {len(tx_symbols)}, remainder = {remainder}")
# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)

while True:
    print("Transmitting...")
    time.sleep(10)




