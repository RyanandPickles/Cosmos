import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys

from cosmos import *
from digicomm import *
from helpers import bits_to_bytes, bytes_to_file

import adi

# Directory for saving plots
dir_plots = 'plots/'

# ---------------------------------------------------------------
# Setup.
# ---------------------------------------------------------------
sdr_rx = adi.Pluto("usb:0.1.5")

rx = PlutoReceiver()
rx.set_sdr(sdr_rx)
rx.set_buffer_size(500e3)
rx.set_channel(7)
rx.set_gain_level(80)
rx.desired_transmit_symbols_real = False
rx.num_transmit_symbols = 200

M=16
k = int(np.log2(M))
rx.num_transmit_symbols = 762 # change these hardcoded values later pls
remainder = 0  # change these hardcoded values later pls
constellation = get_qam_constellation(M, Es=1)


# print(rx.sdr)
# ---------------------------------------------------------------
# Receive.
# ---------------------------------------------------------------
rx_symbols = rx.receive()

rx_bits = qam_symbols_to_bits(rx_symbols, M, remainder)
rx_bytes = bits_to_bytes(rx_bits)
bytes_to_file(rx_bytes, "output/received.rtf")

if True:
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(rx_symbols),np.imag(rx_symbols), color='red', label='Received QAM Symbols')
    plt.title('Data Symbols After Equalization')
    plt.xlabel('Real Component')
    plt.ylabel('Imaginary Component')
    plt.grid(True)
    plt.legend()
    filename = dir_plots + 'main_tx_rx_02' + '.pdf'
    plt.savefig(filename)
    filename = dir_plots + 'main_tx_rx_02' + '.svg'
    plt.savefig(filename)
    plt.show()
