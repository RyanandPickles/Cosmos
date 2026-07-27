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
sdr_tx = adi.Pluto("usb:1.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_channel(7)
tx.set_power_level(95)

# ---------------------------------------------------------------
# Generate random symbols.
# ---------------------------------------------------------------

M=16

filebytes = file_to_bytes("/Users/ryanli/Desktop/Test.rtf")
filebits = bytes_to_bits(filebytes)

tx_symbols, remainder = bits_to_qam_symbols(filebits, M)
constellation = get_qam_constellation(M, Es=1)
print(f"num_transmit_symbols = {len(tx_symbols)}, remainder = {remainder}")
# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)

while True:
    print("Transmitting...")
    time.sleep(10)




