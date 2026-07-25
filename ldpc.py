"""
ldpc.py
-------
A self-contained LDPC (Low-Density Parity-Check) error correction codec
for the Cosmos SDR video project.

This builds directly on the Week 2 Day 4 LDPC notebook: same idea (sparse
parity-check matrix H, iterative bit-flipping decoder for a Binary
Symmetric Channel), but wrapped into reusable functions that plug into
the rest of the pipeline (compression -> LDPC encode -> transmit ->
receive -> LDPC decode -> decompress).

No radio / hardware dependencies here on purpose -- this can be fully
tested locally (see the __main__ block at the bottom) before it ever
touches main_tx.py / main_rx.py.

-------------------------------------------------------------------------
HOW THE CODE IS BUILT (systematic LDPC)
-------------------------------------------------------------------------
We use a "systematic" construction, which keeps things simple and fast
while still being a valid sparse LDPC code:

    A  : (m x k) sparse random binary matrix      (m = n - k)
    H  = [ A | I_m ]                              parity-check matrix (m x n)
    G  = [ I_k | A^T ]                             generator matrix    (k x n)

Given a k-bit message vector `msg`:
    parity   = (A @ msg) mod 2          (m parity bits)
    codeword = [msg | parity]           (n bits total)

This automatically satisfies H @ codeword^T = 0 (mod 2), which is exactly
the property a valid codeword must have.

Rate of the code = k / n (higher rate = less redundancy = less protection,
but more actual data gets through per transmitted bit).
"""

import numpy as np


# ---------------------------------------------------------------
# Code construction
# ---------------------------------------------------------------
def generate_ldpc_matrices(k, m, col_weight=3, seed=None):
    """
    Build a sparse systematic LDPC code.

    Parameters
    ----------
    k : int
        Number of message (data) bits per block.
    m : int
        Number of parity bits per block. n = k + m is the codeword length.
    col_weight : int
        Target number of 1s per column of A (controls sparsity / how
        "LDPC-like" the code is). Higher = denser = usually stronger but
        slower to decode.
    seed : int or None
        RNG seed, for reproducible codes (useful so TX and RX agree on
        the exact same code without transmitting the matrix itself).

    Returns
    -------
    A : (m x k) binary numpy array
    H : (m x n) binary numpy array, parity-check matrix = [A | I_m]
    n : int, codeword length (k + m)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((m, k), dtype=np.uint8)

    for col in range(k):
        weight = min(col_weight, m)
        rows = rng.choice(m, size=weight, replace=False)
        A[rows, col] = 1

    I_m = np.eye(m, dtype=np.uint8)
    H = np.concatenate((A, I_m), axis=1)
    n = k + m
    return A, H, n


# ---------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------
def ldpc_encode_block(msg_bits, A):
    """
    Encode a single k-bit message block into an n-bit codeword.

    msg_bits : 1D array of 0/1, length k
    A        : (m x k) matrix from generate_ldpc_matrices()

    Returns: 1D array of 0/1, length k + m
    """
    msg_bits = np.asarray(msg_bits, dtype=np.uint8)
    parity = (A @ msg_bits) % 2
    codeword = np.concatenate((msg_bits, parity))
    return codeword


def ldpc_encode(bit_string, A, k):
    """
    Encode an arbitrary-length bitstring (e.g. from helpers.jpg_to_bits)
    by splitting it into k-bit blocks and LDPC-encoding each one.

    bit_string : str of '0'/'1' characters, any length
    A          : (m x k) matrix from generate_ldpc_matrices()
    k          : message block size (must match A's column count)

    Returns
    -------
    encoded_bit_string : str of '0'/'1', the full encoded stream
    original_length    : int, length of bit_string before padding
                          (you need this to correctly trim the decoded
                          output back to the original size)
    """
    original_length = len(bit_string)

    # Pad so the input divides evenly into k-bit blocks
    pad_len = (-original_length) % k
    padded = bit_string + '0' * pad_len

    bits = np.array([int(b) for b in padded], dtype=np.uint8)
    blocks = bits.reshape(-1, k)

    encoded_blocks = [ldpc_encode_block(block, A) for block in blocks]
    encoded_bits = np.concatenate(encoded_blocks)

    encoded_bit_string = ''.join(str(b) for b in encoded_bits)
    return encoded_bit_string, original_length


# ---------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------
def ldpc_decode_block(received_bits, H, max_iterations=50):
    """
    Iterative bit-flipping decoder for a single received n-bit block
    (same core idea as the Week 2 Day 4 notebook's BSC decoder).

    received_bits : 1D array of 0/1, length n (possibly with bit errors)
    H             : (m x n) parity-check matrix
    max_iterations: safety cap on decode iterations

    Returns
    -------
    corrected : 1D array of 0/1, length n -- the decoder's best guess
                at the transmitted codeword
    success   : bool, True if all parity checks are satisfied
                (i.e. the decoder believes it fully corrected the block)
    """
    c = np.array(received_bits, dtype=np.uint8).copy()

    for _ in range(max_iterations):
        syndrome = (H @ c) % 2
        if not syndrome.any():
            return c, True

        # For each bit position, count how many *unsatisfied* parity
        # checks it participates in.
        unsatisfied_counts = H.T @ syndrome

        max_count = unsatisfied_counts.max()
        if max_count == 0:
            break

        # Flip every bit tied for the most unsatisfied checks
        flip_idx = np.where(unsatisfied_counts == max_count)[0]
        c[flip_idx] = 1 - c[flip_idx]

    syndrome = (H @ c) % 2
    return c, not syndrome.any()


def ldpc_decode(encoded_bit_string, H, k, n, original_length, max_iterations=50):
    """
    Decode a full encoded bitstream produced by ldpc_encode().

    encoded_bit_string : str of '0'/'1', possibly containing bit errors
    H                   : (m x n) parity-check matrix (must match the H
                          used at TX -- same k, m, seed)
    k, n                : block sizes used at encode time
    original_length     : the original_length returned by ldpc_encode(),
                          used to trim off padding at the end
    max_iterations      : passed through to ldpc_decode_block()

    Returns
    -------
    decoded_bit_string : str of '0'/'1', trimmed back to original_length
    num_blocks_failed  : int, how many blocks the decoder could NOT fully
                          correct (parity checks still failing after
                          max_iterations) -- a useful health metric
    """
    bits = np.array([int(b) for b in encoded_bit_string], dtype=np.uint8)
    blocks = bits.reshape(-1, n)

    decoded_msg_bits = []
    num_blocks_failed = 0

    for block in blocks:
        corrected, success = ldpc_decode_block(block, H, max_iterations)
        if not success:
            num_blocks_failed += 1
        decoded_msg_bits.append(corrected[:k])  # message bits are the first k (systematic)

    decoded_bits = np.concatenate(decoded_msg_bits)
    decoded_bit_string = ''.join(str(b) for b in decoded_bits)
    decoded_bit_string = decoded_bit_string[:original_length]  # trim padding
    return decoded_bit_string, num_blocks_failed


# ---------------------------------------------------------------
# Channel simulation (for local testing without the radios)
# ---------------------------------------------------------------
def transmit_bsc(bit_string, p, seed=None):
    """
    Simulate a Binary Symmetric Channel: flips each bit independently
    with probability p. Same model used in the Week 2 Day 2 / Day 4
    notebooks. Useful for testing this module before touching hardware.
    """
    rng = np.random.default_rng(seed)
    bits = np.array([int(b) for b in bit_string], dtype=np.uint8)
    flips = rng.random(len(bits)) < p
    bits[flips] = 1 - bits[flips]
    return ''.join(str(b) for b in bits)


# ---------------------------------------------------------------
# Local self-test / demo
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Code parameters: rate 1/2 code, 16 data bits + 16 parity bits per block
    k = 16
    m = 16
    n = k + m
    seed = 42  # TX and RX must use the SAME seed to build the SAME code

    A, H, n = generate_ldpc_matrices(k, m, col_weight=3, seed=seed)

    # Random test message
    rng = np.random.default_rng(0)
    test_bits = ''.join(str(b) for b in rng.integers(0, 2, size=400))

    # TX side
    encoded, original_length = ldpc_encode(test_bits, A, k)
    print(f"Original bits: {len(test_bits)}  Encoded bits: {len(encoded)}  Rate: {k/n:.2f}")

    # Simulate a noisy channel
    p = 0.03  # 3% bit flip probability
    received = transmit_bsc(encoded, p, seed=1)
    num_flipped = sum(a != b for a, b in zip(encoded, received))
    print(f"Channel flipped {num_flipped} / {len(encoded)} bits ({p*100:.1f}% target)")

    # RX side
    decoded, num_failed_blocks = ldpc_decode(received, H, k, n, original_length)

    correct = (decoded == test_bits)
    print(f"Blocks that failed to fully correct: {num_failed_blocks} / {len(encoded)//n}")
    print(f"Decoded matches original: {correct}")

    if not correct:
        num_wrong_bits = sum(a != b for a, b in zip(decoded, test_bits))
        print(f"  -> {num_wrong_bits} bits still wrong after decoding")
