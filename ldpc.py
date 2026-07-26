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


def _bitstring_to_uint8(bit_string):
    """Vectorized '0'/'1' str -> np.uint8 array (no per-char Python loop)."""
    return np.frombuffer(bit_string.encode('ascii'), dtype=np.uint8) - ord('0')


def _uint8_to_bitstring(bits):
    """Vectorized np.uint8 (0/1) array -> '0'/'1' str (no per-element join)."""
    return (bits.astype(np.uint8) + ord('0')).tobytes().decode('ascii')


def ldpc_encode(bit_string, A, k):
    """
    Encode an arbitrary-length bitstring (e.g. from helpers.jpg_to_bits)
    by splitting it into k-bit blocks and LDPC-encoding ALL blocks at
    once via a single batched matmul, instead of a Python loop over
    ldpc_encode_block() per block. Same math, same output -- just
    reshaped so numpy does the work in one call instead of thousands.

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

    bits = _bitstring_to_uint8(padded)
    blocks = bits.reshape(-1, k)  # (num_blocks, k)

    # parity[i] = (A @ blocks[i]) % 2 for every block i at once:
    # blocks (num_blocks, k) @ A.T (k, m) -> (num_blocks, m)
    parity = (blocks.astype(np.int64) @ A.T.astype(np.int64)) % 2
    codewords = np.concatenate((blocks, parity.astype(np.uint8)), axis=1)  # (num_blocks, n)

    encoded_bit_string = _uint8_to_bitstring(codewords.reshape(-1))
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


def ldpc_decode_blocks_batched(blocks, H, max_iterations=50):
    """
    Same bit-flipping algorithm as ldpc_decode_block(), but runs ALL
    blocks through the iteration loop together as one (num_blocks, n)
    matrix, so each of the (up to) max_iterations rounds is a couple of
    matmuls over every block at once instead of a Python-level loop
    doing one block at a time. Blocks are decoded independently of each
    other (H is shared, but nothing crosses between rows), so batching
    like this gives bit-for-bit identical results to calling
    ldpc_decode_block() on each row separately -- just much faster.

    blocks : (num_blocks, n) uint8 array of received bits
    H      : (m, n) parity-check matrix
    Returns: (corrected (num_blocks, n) uint8 array, success (num_blocks,) bool array)
    """
    C = blocks.astype(np.int64).copy()
    Ht = H.T.astype(np.int64)  # (n, m), reused every iteration
    num_blocks = C.shape[0]
    active = np.ones(num_blocks, dtype=bool)  # rows not yet converged/stuck

    for _ in range(max_iterations):
        if not active.any():
            break

        syndrome = (C[active] @ Ht) % 2  # (num_active, m)
        row_done = ~syndrome.any(axis=1)

        # Mark newly-converged rows as inactive; keep working on the rest.
        active_idx = np.where(active)[0]
        active[active_idx[row_done]] = False
        still_going = ~row_done
        if not still_going.any():
            continue

        syndrome = syndrome[still_going]
        rows = active_idx[still_going]

        unsatisfied_counts = syndrome @ H  # (num_still_going, n)
        max_count = unsatisfied_counts.max(axis=1)

        stuck = max_count == 0
        if stuck.any():
            active[rows[stuck]] = False  # can't improve further -> stop, will report as failed below

        moving = ~stuck
        if moving.any():
            flip_mask = unsatisfied_counts[moving] == max_count[moving][:, None]
            C[rows[moving]] ^= flip_mask.astype(np.int64)

    final_syndrome = (C @ Ht) % 2
    success = ~final_syndrome.any(axis=1)
    return C.astype(np.uint8), success


def ldpc_decode(encoded_bit_string, H, k, n, original_length, max_iterations=50):
    """
    Decode a full encoded bitstream produced by ldpc_encode(), batching
    all blocks through ldpc_decode_blocks_batched() instead of looping
    over ldpc_decode_block() one block at a time.

    encoded_bit_string : str of '0'/'1', possibly containing bit errors
    H                   : (m x n) parity-check matrix (must match the H
                          used at TX -- same k, m, seed)
    k, n                : block sizes used at encode time
    original_length     : the original_length returned by ldpc_encode(),
                          used to trim off padding at the end
    max_iterations      : passed through to the decoder

    Returns
    -------
    decoded_bit_string : str of '0'/'1', trimmed back to original_length
    num_blocks_failed  : int, how many blocks the decoder could NOT fully
                          correct (parity checks still failing after
                          max_iterations) -- a useful health metric
    """
    bits = _bitstring_to_uint8(encoded_bit_string)
    blocks = bits.reshape(-1, n)

    corrected, success = ldpc_decode_blocks_batched(blocks, H, max_iterations)
    num_blocks_failed = int((~success).sum())

    decoded_bits = corrected[:, :k].reshape(-1)  # message bits are the first k (systematic) of each block
    decoded_bit_string = _uint8_to_bitstring(decoded_bits)
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
