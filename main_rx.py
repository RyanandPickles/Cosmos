import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys
import os
from datetime import datetime


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
rx.set_buffer_size(1e6)
rx.set_channel(7)
rx.set_gain_level(80)
rx.desired_transmit_symbols_real = False

M=16
k = int(np.log2(M))
header_bits = 32
max_bits=7968

rx.num_transmit_symbols = (header_bits+max_bits) // k 

constellation = get_qam_constellation(M, Es=1)


# print(rx.sdr)
# ---------------------------------------------------------------
# Receive.
# ---------------------------------------------------------------
rx_symbols = rx.receive()

rx_bits = qam_symbols_to_bits(rx_symbols, M, 0)

header_bit_values = rx_bits[:header_bits]
header_string = "".join(str(b) for b in header_bit_values)
message_len = int(header_string, 2)

message_bits = rx_bits[header_bits: header_bits + message_len]

rx_bytes = bits_to_bytes(message_bits)

output_dir = "/Users/vincent/Desktop/SDRReceivedLogs"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"rx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
output_path = os.path.join(output_dir, output_filename)

bytes_to_file(rx_bytes, output_path)

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
