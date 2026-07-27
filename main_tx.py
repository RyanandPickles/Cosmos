import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys

from cosmos import *
from digicomm import *
from functions import file_to_bytes, bytes_to_bits

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

filebytes = file_to_bytes("file path remember to change this btw")
filebits = bytes_to_bits(filebytes)

tx_symbols, remainder = bits_to_qam_symbols(filebits, M)
constellation = get_qam_constellation(M, Es=1)

# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)

while True:
    print("Transmitting...")
    time.sleep(10)




