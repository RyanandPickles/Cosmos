#used for reading arguments, used to mark info as variables
import argparse
#used for filtering the contentso f a folder down to matching pattern
import glob
#packing and unpacking binary headers
import struct
import time
#used for error detection checksums
import zlib
import numpy as np
import adi
from cosmos import PlutoTransmitter
from digicomm import bits_to_qam_symbols
from helpers import jpg_to_bits
from ldpc import LDPCCode, bitstring_to_uint8

#setup
LDPC_K = 256
LDPC_M = 256
LDPC_COL_WEIGHT = 3
LDPC_SEED = 42 #used for identical random code
QAM_ORDER = 16
FRAME_PAYLOAD_BYTES = 2048
HEADER_BYTES = 8

SDR_TX_URI = "usb:1.12.5"
SAMPLE_RATE = 6_000_000
CHANNEL = 7
TX_POWER_LEVEL = 95

def compute_num_tx_symbols(code, qam_order, frame_payload_bytes):
    """
Description:
    Computes how many QAM symbols one frame turns into from LDPC code and QAM order   
Parameters:
Returns: 
    """
    frame_bits = frame_payload_bytes * 8
    num_blocks = -(-frame_bits // code.K) #ceiling
    total_encoded_bits = num_blocks * code.n
    bits_per_symbol = int(np.log2(qam_order))
    num_qam_symbols = -(-total_encoded_bits // bits_per_symbol) #ceiling
    return num_qam_symbols, total_encoded_bits

def pack_frame(jpg_bytes, payload_size):
    """
Description:
    raw jpg bytes are converted to [length][data][crc32][zero-padding]
Parameters:
Returns: breh im too lazy
    """
    data_capacity = payload_size - HEADER_BYTES
    if len(jpg_bytes) > data_capacity:
        raise ValueError(f"jpg size {len(jpg_bytes)} exceeds {data_capcity} bytes")
    #calculates cyclic reduncancy check for error detection, 32 bits
    crc = zlib.crc32(jpg_bytes) & 0xFFFFFFFF
    header = struct.pack('>I', len(jpg_bytes))
    footer = struct.pack('>I', crc)
    pad = bytes(data_capacity - len(jpg_bytes))
    return header + jpg_bytes + footer + pad