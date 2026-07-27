import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys

from cosmos import *
from digicomm import *

import adi

# Directory for saving plots
dir_plots = 'plots/'

# ---------------------------------------------------------------
# Setup.
# --------------------------------------3-------------------------
sdr_tx = adi.Pluto("usb:1.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_channel(9)
tx.set_power_level(95)

# ---------------------------------------------------------------
# Generate random symbols.
# ---------------------------------------------------------------
num_symbols = 200 # number of random data symbols to generate
M=16
tx_symbols, const = gen_rand_qam_symbols(num_symbols, M=M)
# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)

while True:
    print("Transmitting...")
    time.sleep(10)




