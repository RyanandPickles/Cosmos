import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys
import adi

from cosmos import *
from digicomm import *

from helpers import *

# ---------------------------------------------------------------
# Setup.
# --------------------------------------3-------------------------
sdr_tx = adi.Pluto("usb:0.1.5")

tx = PlutoTransmitter()
tx.set_sdr(sdr_tx)
tx.set_channel(7)
tx.set_power_level(75)

#reads image as bytes
file_path = "replaceWithImageName.jpg"  
with open(file_path, "rb") as f:
    raw_bytes = f.read()

#lz4 compress bytes and convert to bits
tx_bits, compressed_len = bytes_to_lz4_bits(raw_bytes)
print(f"Original size  : {len(raw_bytes)} bytes")
print(f"Compressed size: {compressed_len} bytes")
print(f"Total bits sent: {len(tx_bits)} bits")

# bits to 2 PAM
tx_symbols = np.where(tx_bits == 1, 1.0, -1.0)

# transmit over Pluto
tx.transmit(tx_symbols)

while True:
    print("Transmitting LZ4 compressed PAM data...")
    time.sleep(10)
