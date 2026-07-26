"""
transport.py
-------------
Glue layer between compression.py (video frames <-> bytes) and the
cosmos.py / digicomm.py radio PHY (which speaks in QAM symbol arrays).

WHY A FIXED-SIZE FRAME
------------------------
PlutoReceiver.receive() needs `rx.num_transmit_symbols` set *before*
calling receive() -- there's no in-band header telling the receiver how
long the payload is, so both sides must agree on exactly how many QAM
symbols make up one video frame, ahead of time. That's why
compression.py adaptively compresses each frame to fit a fixed byte
budget: it lets every frame turn into exactly the same number of LDPC
codeword bits, and therefore exactly the same number of QAM symbols,
every single time.

Per-frame bit budget, end to end:

    num_transmit_symbols (radio) --x bits_per_symbol (QAM order)-->
    codeword_bits --split into LDPC blocks of size n = k + m-->
    payload_bits = num_blocks * k --minus HEADER_BITS-->
    max_payload_bytes  <-- this is the number compression.py targets

LinkConfig below does that arithmetic once, so send.py / receive.py
just build one LinkConfig and call encode_for_tx() / decode_from_rx().

FRAME FORMAT (before LDPC, "payload_bits" long)
--------------------------------------------------
[ 32-bit length (bytes) | 32-bit CRC32 of the JPEG bytes | JPEG bytes | zero pad ]

The length+CRC header lets the receiver tell a genuinely-good frame
apart from one the LDPC decoder *thinks* it fixed but didn't (LDPC's
own "success" flag only means "all parity checks passed" -- for a
weak/sparse code with a lot of bit errors it can converge on the wrong
codeword and still report success, so a real integrity check is
important, especially for live video where showing a corrupted frame
is worse than skipping it).
"""

import zlib
import numpy as np

from digicomm import get_qam_constellation
from ldpc import generate_ldpc_matrices, ldpc_encode, ldpc_decode

import compression

HEADER_LEN_BITS = 32
HEADER_CRC_BITS = 32
HEADER_BITS = HEADER_LEN_BITS + HEADER_CRC_BITS


class LinkConfig:
    """
    Precomputes and stores everything both send.py and receive.py need
    to agree on. Build ONE of these with the SAME parameters on both
    the TX and RX computers.
    """

    def __init__(self, num_transmit_symbols, M=16, ldpc_k=128, ldpc_m=128,
                 ldpc_col_weight=3, ldpc_seed=42):
        bits_per_symbol = int(np.log2(M))
        if 2 ** bits_per_symbol != M:
            raise ValueError("M must be a power of 2.")

        n = ldpc_k + ldpc_m
        codeword_bits = num_transmit_symbols * bits_per_symbol
        num_blocks = codeword_bits // n
        if num_blocks < 1:
            raise ValueError(
                f"num_transmit_symbols={num_transmit_symbols} is too small for "
                f"one LDPC block (n={n} bits needs {n // bits_per_symbol} symbols)."
            )
        used_codeword_bits = num_blocks * n
        if used_codeword_bits != codeword_bits:
            # Trim to a whole number of LDPC blocks so we never have partial
            # blocks/symbols. This slightly reduces num_transmit_symbols;
            # store the corrected value so TX/RX agree on the real one.
            usable_symbols = used_codeword_bits // bits_per_symbol
            print(f"Warning: {num_transmit_symbols} symbols isn't a whole number of "
                  f"LDPC blocks at M={M}. Using {usable_symbols} symbols/frame instead.")
            num_transmit_symbols = usable_symbols

        payload_bits = num_blocks * ldpc_k
        if payload_bits <= HEADER_BITS:
            raise ValueError("payload_bits too small to even fit the frame header; "
                              "increase num_transmit_symbols or decrease ldpc_k/ldpc_m.")

        self.M = M
        self.bits_per_symbol = bits_per_symbol
        self.ldpc_k = ldpc_k
        self.ldpc_m = ldpc_m
        self.n = n
        self.num_blocks = num_blocks
        self.payload_bits = payload_bits
        self.codeword_bits = used_codeword_bits
        self.num_transmit_symbols = num_transmit_symbols
        self.max_payload_bytes = (payload_bits - HEADER_BITS) // 8

        self.A, self.H, n_check = generate_ldpc_matrices(
            ldpc_k, ldpc_m, col_weight=ldpc_col_weight, seed=ldpc_seed)
        assert n_check == n

        self.constellation = get_qam_constellation(M, Es=1)

    def __repr__(self):
        return (f"LinkConfig(M={self.M}, symbols/frame={self.num_transmit_symbols}, "
                f"LDPC k={self.ldpc_k} m={self.ldpc_m} rate={self.ldpc_k/self.n:.2f}, "
                f"max_payload_bytes={self.max_payload_bytes})")


# ---------------------------------------------------------------
# Frame packing / unpacking (bytes <-> fixed-length bit string)
# ---------------------------------------------------------------
def pack_frame(jpg_bytes, config):
    """JPEG bytes -> fixed-length ('0'/'1') bit string, ready for ldpc_encode."""
    if len(jpg_bytes) > config.max_payload_bytes:
        raise ValueError(
            f"Frame is {len(jpg_bytes)} bytes, budget is {config.max_payload_bytes} bytes. "
            f"compression.py should have compressed to fit -- check its max_bytes argument."
        )

    length_field = format(len(jpg_bytes), f'0{HEADER_LEN_BITS}b')
    crc_field = format(zlib.crc32(jpg_bytes) & 0xFFFFFFFF, f'0{HEADER_CRC_BITS}b')
    payload_bits = compression.bytes_to_bits(jpg_bytes)

    bit_string = length_field + crc_field + payload_bits
    pad_len = config.payload_bits - len(bit_string)
    bit_string += '0' * pad_len
    return bit_string


def unpack_frame(bit_string, config):
    """
    Fixed-length bit string -> JPEG bytes, or None if the header/CRC says
    the frame is corrupted (i.e. don't trust it -- caller should hold the
    last good frame instead of displaying garbage).
    """
    length = int(bit_string[0:HEADER_LEN_BITS], 2)
    crc_expected = int(bit_string[HEADER_LEN_BITS:HEADER_BITS], 2)

    if length < 0 or length > config.max_payload_bytes:
        return None  # header itself is implausible -> definitely corrupted

    payload_bits = bit_string[HEADER_BITS:HEADER_BITS + length * 8]
    if len(payload_bits) < length * 8:
        return None

    jpg_bytes = compression.bits_to_bytes(payload_bits)
    crc_actual = zlib.crc32(jpg_bytes) & 0xFFFFFFFF
    if crc_actual != crc_expected:
        return None

    return jpg_bytes


# ---------------------------------------------------------------
# Bits <-> QAM symbols
# ---------------------------------------------------------------
def bits_to_symbols(bit_string, config):
    """
    '0'/'1' string (length must be a multiple of bits_per_symbol) -> complex
    QAM symbol array, using config.constellation. The constellation index
    for a group of bits is just that group's binary value (MSB first) --
    get_qam_constellation() already bakes the Gray-code mapping into the
    layout of the array itself, so a plain binary index is correct here.
    """
    k = config.bits_per_symbol
    bits = np.frombuffer(bit_string.encode('ascii'), dtype=np.uint8) - ord('0')
    bits = bits.reshape(-1, k)
    weights = (2 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    idx = bits @ weights
    return config.constellation[idx]


def symbols_to_bits(symbols, config):
    """Complex QAM symbol array -> '0'/'1' string (nearest-neighbor demod)."""
    const = config.constellation  # shape (M,)
    # distance from every received symbol to every constellation point
    dist = np.abs(symbols[:, None] - const[None, :])
    idx = np.argmin(dist, axis=1)
    k = config.bits_per_symbol
    bits_matrix = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1)
    return ''.join(str(b) for b in bits_matrix.flatten())


# ---------------------------------------------------------------
# Top-level: JPEG bytes <-> transmit-ready QAM symbols
# ---------------------------------------------------------------
def encode_for_tx(jpg_bytes, config):
    """JPEG bytes for one frame -> complex QAM symbol array ready for tx.transmit()."""
    payload_bits = pack_frame(jpg_bytes, config)
    encoded_bits, _ = ldpc_encode(payload_bits, config.A, config.ldpc_k)
    assert len(encoded_bits) == config.codeword_bits
    symbols = bits_to_symbols(encoded_bits, config)
    assert len(symbols) == config.num_transmit_symbols
    return symbols


def decode_from_rx(symbols, config, max_iterations=50):
    """
    Complex QAM symbol array from rx.receive() -> (jpg_bytes_or_None, info)

    info is a dict with 'num_blocks_failed' (from the LDPC decoder) and
    'crc_ok' (whether the header/CRC integrity check passed) so send/receive
    loops can log link quality.
    """
    encoded_bits = symbols_to_bits(symbols, config)
    payload_bits, num_blocks_failed = ldpc_decode(
        encoded_bits, config.H, config.ldpc_k, config.n,
        original_length=config.payload_bits, max_iterations=max_iterations)
    jpg_bytes = unpack_frame(payload_bits, config)
    info = {'num_blocks_failed': num_blocks_failed, 'crc_ok': jpg_bytes is not None}
    return jpg_bytes, info


# ---------------------------------------------------------------
# Throughput / timing helpers
# ---------------------------------------------------------------
# Estimated preamble length in symbols, using PlutoTransmitter's default
# STF/LTF/pilot settings (64*19 STF + 100 zero + 2*937 LTF + 100 zero + 100
# pilots). Recompute this yourself if you change tx.set_stf()/set_ltf().
DEFAULT_PREAMBLE_SYMBOLS = 64 * 19 + 100 + 2 * 937 + 100 + 100  # = 3390


def estimate_frame_time_sec(config, sample_rate, sps=10,
                             preamble_symbols=DEFAULT_PREAMBLE_SYMBOLS):
    """
    Wall-clock time to transmit ONE frame's worth of samples (preamble +
    payload) at a given SDR sample_rate. This is the real limit on
    achievable fps -- it's set by the radio's symbol rate
    (sample_rate / sps), not by how fast compression.py can run.
    """
    total_symbols = preamble_symbols + config.num_transmit_symbols
    total_samples = total_symbols * sps
    return total_samples / sample_rate


def max_achievable_fps(config, sample_rate, sps=10,
                        preamble_symbols=DEFAULT_PREAMBLE_SYMBOLS):
    return 1.0 / estimate_frame_time_sec(config, sample_rate, sps, preamble_symbols)


def recommended_rx_buffer_size(config, sps=10,
                                preamble_symbols=DEFAULT_PREAMBLE_SYMBOLS, margin=2.5):
    """
    Suggested rx.set_buffer_size() -- needs enough raw samples to contain
    the whole frame (preamble + payload) PLUS slack for the frame-sync
    search window to actually find the STF/LTF peak. Capped at the
    PlutoReceiver hardware/software max of 5e6 samples.
    """
    total_symbols = preamble_symbols + config.num_transmit_symbols
    size = int(total_symbols * sps * margin)
    return min(size, int(5e6))


if __name__ == "__main__":
    # Local, hardware-free self-test: compress a synthetic frame, push it
    # through LDPC + QAM (mod) with simulated channel noise, demod + LDPC
    # decode, and confirm we get the same JPEG bytes back (or a clean
    # "corrupted, discard" signal when noise is too high).
    from digicomm import cgauss_rv

    config = LinkConfig(num_transmit_symbols=20000, M=16, ldpc_k=128, ldpc_m=128)
    print(config)

    rng = np.random.default_rng(1)
    fake_frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    jpg, q, size_used = compression.compress_to_budget(
        fake_frame, config.max_payload_bytes, width=320, height=240)
    print(f"Compressed frame: {len(jpg)} bytes / budget {config.max_payload_bytes} bytes, quality={q}")

    symbols = encode_for_tx(jpg, config)
    print(f"Modulated to {len(symbols)} symbols (config expects {config.num_transmit_symbols})")

    # No-noise round trip
    recovered, info = decode_from_rx(symbols, config)
    assert recovered == jpg, "no-noise round trip failed"
    print(f"No-noise round trip OK. info={info}")

    # Noisy round trip (simulate a weak but usable channel)
    noise = cgauss_rv(0 + 0j, 0.02, len(symbols))
    noisy_symbols = symbols + noise
    recovered_noisy, info_noisy = decode_from_rx(noisy_symbols, config)
    ok = (recovered_noisy == jpg)
    print(f"Noisy round trip: match={ok}, info={info_noisy}")

    print("transport.py self-test passed.")
