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
    #length of jpg is in 4 raw bytes
    header = struct.pack('>I', len(jpg_bytes))
    #crc value is 4 bytes
    footer = struct.pack('>I', crc)
    #zero padding which fills the rest of the frame
    pad = bytes(data_capacity - len(jpg_bytes))
    #final format looks like this: [ 4 bytes: length ][ jpg_bytes: N bytes ][ 4 bytes: CRC ][ pad: zeros ]
    return header + jpg_bytes + footer + pad

def bytes_to_bitstring(data):
    #joins 8 bit binary chunks into one text string
    return ''.join(format(b,'08b') for b in data)

def main():
    #creates the parser object 
    parser = argparse.ArgumentParser(description="stream jpg frames")
    #adds the argument which allows parser to knokw that the --frames option exists
    parser.add_argument('--frames', default='frames/*.jpg', help="glob pattern for jpg frames")
    #the fps option is also added
    parser.add_argument('--fps', type=float, default=10.0)
    #variable that stores args.frames and args.fps
    args = parser.parse_args()

######### downdowndowndowndowndowndowndowndowndowndowndown

    code = LDPCCode(k=LDPC_K, m=LDPC_M, column_weight=LDPC_COL_WEIGHT, seed=LDPC_SEED)
    num_tx_symbols, total_encoded_bits = compute_num_tx_symbols(code, QAM_ORDER, FRAME_PAYLOAD_BYTES)
    print(f"LDPC rate: {code.k}/{code.n} ({code.k/code.n:.2f})" 
          f"Frame payload: {FRAME_PAYLOAD_BYTES} bytes --> {num_tx_symbols} QAM symbols/frame"
          )
##########^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    sdr_tx = adi.Pluto(SDR_TX_URI)
    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_sample_rate(SAMPLE_RATE)
    tx.set_sample_rate(CHANNEL)
    tx.set_power_level(TX_POWER_LEVEL)

######### downdowndowndowndowndowndowndowndowndowndowndown

    frame_paths = sorted(glob.glob(args.frames))
    if not frame_paths:
        raise FileNotFoundError(f"No jpg frames found matching {args.frames}")
    print (f"Streaming {len(frame_paths)} frames from '{args.frames}' at target {args.fps} fps")

    frame_interval = 1.0 /args.fps
    frame_index=0

    while True:
        t_start = time.time()

        path = frame_paths[frame_index % len(frame_paths)]
        frame_index +=1

        with open(path, 'rb') as f:
            jpg_bytes = f.read()
        packed = pack_frame(jpg_bytes, FRAME_PAYLOAD_BYTES)
        bitstring = bytes_to_bitstring(packed)

        encoded_bits, original_length = code.encode(bitstring)

        bits_array = bitstring_to_uint8(encoded_bits)
        qam_symbols, _ = bits_to_qam_symbols(bits_array, QAM_ORDER)

        if len(qam_symbols) < num_tx_symbols:
            qam_symbols = np.concatenate([qam_symbols, np.zeros(num_tx_symbols - len(qam_symbols), dtype=qam_symbols.dtype)])
        elif len(qam_symbols) > num_tx_symbols:
            qam_symbols = qam_symbols[:num_tx_symbols]

        tx.transmit(qam_symbols)

        elapsed = time.time() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print (f"[framce {frame_index}] encode transmit took {elapsed*1000:.1f} ms")


if __name__ == "__main__":
    main()