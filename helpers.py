import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import random
import zlib

from cryptography.fernet import Fernet
def get_random_integers(n, m, seed):
    """
    Generates n random integers between 0 and m-1 
    using a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    return [rng.randrange(m) for _ in range(n)]

def clamp(n, smallest, largest):
    if n < smallest:
        print(f"Warning: {n} is below the permitted minimum. Clamping to {smallest}.")
        return smallest
    if n > largest:
        print(f"Warning: {n} is above the permitted maximum. Clamping to {largest}.")
        return largest
    return n

def map_level(level, lower, upper, resolution):
    """
    Maps a level (0-100) to a range [lower, upper] 
    with a specific step resolution.
    """
    # Clamp level to 0-100
    level = max(0, min(100, level))
    
    # Calculate linear interpolation
    mapped = lower + (level / 100) * (upper - lower)
    
    # Apply resolution (snap to nearest increment)
    result = round(mapped / resolution) * resolution
    
    return result

def jpg_to_bits(file_path):
    with open(file_path, 'rb') as f:
        # Read the file as binary
        binary_data = f.read()
    
    # Convert binary data to a bit sequence (a string of '0's and '1's)
    bit_sequence = ''.join(format(byte, '08b') for byte in binary_data)
    return bit_sequence

def bits_to_jpg(bit_sequence, output_path):
    # Convert the bit sequence back into bytes
    byte_data = bytes(int(bit_sequence[i:i+8], 2) for i in range(0, len(bit_sequence), 8))
    
    # Write the byte data to a new file
    with open(output_path, 'wb') as f:
        f.write(byte_data)

def unpack_jpg_bits(image_path):
    try:
        with open(image_path, 'rb') as file:
            image_data = file.read()
            bits = ''.join(format(byte, '08b') for byte in image_data)
            return bits
    except FileNotFoundError:
        return "Error: File not found."

def file_to_bytes(file_path):
    with open(file_path, "rb") as file:
        file_contents = file.read()
    return file_contents

def bytes_to_bits(byte_data):
    byte_real = np.frombuffer(byte_data, dtype = np.uint8)
    bits = np.unpackbits(byte_real, bitorder="big")
    return bits

def bits_to_bytes(bits):
    byte_real = np.packbits(bits, bitorder="big")
    byte_data = byte_real.tobytes()
    return byte_data

def bytes_to_file(byte_data, output_file_path):
    with open(output_file_path, "wb") as file:
        file.write(byte_data)

def encrypt_file(info, key):
    fernet = Fernet(key)
    return fernet.encrypt(info)

def decrypt_file(encrypted_info, key):
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_info)

def compress_file(info: bytes) -> bytes:
    return zlib.compress(info, level=9)

def decompress_file(info: bytes) -> bytes:
    return zlib.decompress(info)

try:
    from pyldpc import make_ldpc, decode as _pyldpc_decode, get_message as _pyldpc_get_msg
    _LDPC_AVAILABLE = True
except ImportError:
    _LDPC_AVAILABLE = False

# These constants must be identical on TX and RX.
# n=504, d_v=3, d_c=6  →  rate = 1 - d_v/d_c = 0.5
# n·d_v / d_c = 252  (integer ✓)  →  k = n - 252 = 252 info bits per block
_LDPC_N    = 504
_LDPC_D_V  = 3
_LDPC_D_C  = 6
_LDPC_SEED = 42

_ldpc_H = None
_ldpc_G = None
_ldpc_k = None   # info bits per block (252 with the params above)


def _init_ldpc() -> None:
    """Build LDPC matrices once and cache them. ~1-2 s first call."""
    global _ldpc_H, _ldpc_G, _ldpc_k
    if not _LDPC_AVAILABLE:
        raise ImportError("Run:  pip install pyldpc")
    print("Initialising LDPC matrices (one-time) …")
    _ldpc_H, _ldpc_G = make_ldpc(
        _LDPC_N, _LDPC_D_V, _LDPC_D_C,
        systematic=True, sparse=False, seed=_LDPC_SEED,
    )
    _ldpc_k = int(_ldpc_G.shape[1])   # 252
    print(f"LDPC ready: n={_LDPC_N}, k={_ldpc_k}, rate={_ldpc_k/_LDPC_N:.2f}")


def ldpc_encode(bits: np.ndarray) -> np.ndarray:
    """
    Rate-1/2 block-LDPC encode.

    Input : flat binary array (any length)
    Output: coded bits — roughly 2× longer.
    Pads input to the next multiple of k before encoding.
    """
    if _ldpc_G is None:
        _init_ldpc()

    G, k, n = _ldpc_G, _ldpc_k, _LDPC_N

    # Pad to a multiple of k
    pad = (-len(bits)) % k
    bits_padded = np.concatenate([bits.astype(int), np.zeros(pad, dtype=int)])
    n_blocks = len(bits_padded) // k

    coded = np.empty(n_blocks * n, dtype=int)
    for i in range(n_blocks):
        v = bits_padded[i*k : (i+1)*k]
        coded[i*n : (i+1)*n] = G.dot(v) % 2   # GF(2) systematic encoding

    return coded


def ldpc_decode(coded_bits: np.ndarray, snr_db: float = 5.0,
                maxiter: int = 100) -> np.ndarray:
    """
    Belief-propagation LDPC decode from hard QAM decisions.

    coded_bits : hard 0/1 bits straight from qam_symbols_to_bits()
    snr_db     : assumed channel SNR fed to BP channel-reliability init.
                 Tune this if you see excess errors: try 3–8 dB.
    maxiter    : BP iterations per block (higher = better, slower).
    """
    if _ldpc_H is None:
        _init_ldpc()

    H, G, k, n = _ldpc_H, _ldpc_G, _ldpc_k, _LDPC_N
    n_blocks = len(coded_bits) // n

    decoded = np.empty(n_blocks * k, dtype=int)
    for i in range(n_blocks):
        block = coded_bits[i*n : (i+1)*n].astype(float)
        # Hard bits → soft BPSK values:  0 → +1,  1 → −1
        bpsk = 1.0 - 2.0 * block
        d = _pyldpc_decode(H, bpsk, snr_db, maxiter=maxiter)
        decoded[i*k : (i+1)*k] = _pyldpc_get_msg(G, d)

    return decoded


def ldpc_coded_symbol_count(n_info_bits: int, bits_per_symbol: int) -> int:
    """
    How many QAM symbols the LDPC-coded payload occupies.
    Call this once at startup to set num_transmit_symbols on the RX.

    With n=504, k=252, and 8000 info bits:
      ceil(8000/252) = 32 blocks × 504 = 16 128 coded bits
      16 128 / 4 bits/sym = 4 032 QAM-16 symbols   (vs 2 000 without LDPC)
    """
    if _ldpc_k is None:
        _init_ldpc()
    n_blocks = -(-n_info_bits // _ldpc_k)   # ceiling division
    return -(-( n_blocks * _LDPC_N) // bits_per_symbol)
