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
sdr_tx = adi.Pluto("usb:0.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_channel(9)
tx.set_power_level(75)

# ---------------------------------------------------------------
# Generate random symbols.
# ---------------------------------------------------------------
num_pam_symbols = 200 # number of random data symbols to generate
tx_symbols = 2*np.random.randint(0,2,num_pam_symbols) - 1
tx_symbols = np.real(tx_symbols)

# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)

while True:
    print("Transmitting...")
    time.sleep(10)




