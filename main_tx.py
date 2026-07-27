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
# ---------------------------------------------------------------
# KEEP YOUR OWN WORKING ADDRESS HERE -- don't overwrite it with this
# placeholder, use whatever's already working for you in send.py.
sdr_tx = adi.Pluto("usb:1.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_sample_rate(6_000_000)
tx.set_channel(7)
tx.set_power_level(95)

# ---------------------------------------------------------------
# Generate random 16-QAM symbols (genuinely complex, unlike the
# earlier real-PAM test -- this exercises both I and Q).
# ---------------------------------------------------------------
num_qam_symbols = 8192  # matches send.py's actual symbol count
tx_symbols, _ = gen_rand_qam_symbols(num_qam_symbols, M=16)

# ---------------------------------------------------------------
# Transmit.
# ---------------------------------------------------------------
tx.transmit(tx_symbols)

while True:
    print("Transmitting...")
    time.sleep(10)
