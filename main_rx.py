import sys
import time
import adi
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from cosmos import *
from digicomm import *
from helpers import *

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
sdr_rx = adi.Pluto("usb:1.11.5")

rx = PlutoReceiver()
rx.set_sdr(sdr_rx)
rx.set_buffer_size(500e3)
rx.set_channel(9)
rx.set_gain_level(80)
rx.desired_transmit_symbols_real = True

# MUST match the number of compressed bits transmitted from main_tx.py
rx.num_transmit_symbols = 12000  #now matches len(tx_bits)

# ---------------------------------------------------------------
# Receive & Demodulate
# ---------------------------------------------------------------
rx_symbols = rx.receive()  # Equalized PAM symbols from PlutoReceiver

#real PAM symbols back to bits
rx_bits = np.where(rx_symbols > 0, 1, 0).astype(np.uint8)

#decompress LZ4 back into bytes
decompressed_bytes = lz4_bits_to_bytes(rx_bits)

if decompressed_bytes is not None:
    print(
        f"Successfully received and decompressed {len(decompressed_bytes)} bytes."
    )

    #save reconstructed output file
    with open("received_output.jpg", "wb") as f:
        f.write(decompressed_bytes)
else:
    print("Failed to decompress frame due to bit errors.")

# Plot Constellation Plot
if True:
    plt.figure(figsize=(6, 6))
    plt.scatter(
        np.real(rx_symbols),
        np.imag(rx_symbols),
        color="red",
        label="Received PAM symbols",
    )
    plt.title("Equalized PAM symbols (LZ4 compressed)")
    plt.xlabel("Real component")
    plt.ylabel("Imaginary component")
    plt.grid(True)
    plt.legend()
    plt.show()
