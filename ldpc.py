import numpy as np


def generate_ldpc_matrices(k, m, col_weight=3, seed=None):
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


def ldpc_encode_block(msg_bits, A):
    msg_bits = np.asarray(msg_bits, dtype=np.uint8)
    parity = (A @ msg_bits) % 2
    codeword = np.concatenate((msg_bits, parity))
    return codeword


def ldpc_encode(bit_string, A, k):
    original_length = len(bit_string)

    pad_len = (-original_length) % k
    padded = bit_string + '0' * pad_len

    bits = np.array([int(b) for b in padded], dtype=np.uint8)
    blocks = bits.reshape(-1, k)

    encoded_blocks = [ldpc_encode_block(block, A) for block in blocks]
    encoded_bits = np.concatenate(encoded_blocks)

    encoded_bit_string = ''.join(str(b) for b in encoded_bits)
    return encoded_bit_string, original_length


def ldpc_decode_block(received_bits, H, max_iterations=50):
    c = np.array(received_bits, dtype=np.uint8).copy()

    for _ in range(max_iterations):
        syndrome = (H @ c) % 2
        if not syndrome.any():
            return c, True

        unsatisfied_counts = H.T @ syndrome

        max_count = unsatisfied_counts.max()
        if max_count == 0:
            break

        flip_idx = np.where(unsatisfied_counts == max_count)[0]
        c[flip_idx] = 1 - c[flip_idx]

    syndrome = (H @ c) % 2
    return c, not syndrome.any()


def ldpc_decode(encoded_bit_string, H, k, n, original_length, max_iterations=50):
    bits = np.array([int(b) for b in encoded_bit_string], dtype=np.uint8)
    blocks = bits.reshape(-1, n)

    decoded_msg_bits = []
    num_blocks_failed = 0

    for block in blocks:
        corrected, success = ldpc_decode_block(block, H, max_iterations)
        if not success:
            num_blocks_failed += 1
        decoded_msg_bits.append(corrected[:k])

    decoded_bits = np.concatenate(decoded_msg_bits)
    decoded_bit_string = ''.join(str(b) for b in decoded_bits)
    decoded_bit_string = decoded_bit_string[:original_length]
    return decoded_bit_string, num_blocks_failed


def transmit_bsc(bit_string, p, seed=None):
    rng = np.random.default_rng(seed)
    bits = np.array([int(b) for b in bit_string], dtype=np.uint8)
    flips = rng.random(len(bits)) < p
    bits[flips] = 1 - bits[flips]
    return ''.join(str(b) for b in bits)


if __name__ == "__main__":
    k = 16
    m = 16
    n = k + m
    seed = 42

    A, H, n = generate_ldpc_matrices(k, m, col_weight=3, seed=seed)

    rng = np.random.default_rng(0)
    test_bits = ''.join(str(b) for b in rng.integers(0, 2, size=400))

    encoded, original_length = ldpc_encode(test_bits, A, k)
    print(f"Original bits: {len(test_bits)}  Encoded bits: {len(encoded)}  Rate: {k/n:.2f}")

    p = 0.03
    received = transmit_bsc(encoded, p, seed=1)
    num_flipped = sum(a != b for a, b in zip(encoded, received))
    print(f"Channel flipped {num_flipped} / {len(encoded)} bits ({p*100:.1f}% target)")

    decoded, num_failed_blocks = ldpc_decode(received, H, k, n, original_length)

    correct = (decoded == test_bits)
    print(f"Blocks that failed to fully correct: {num_failed_blocks} / {len(encoded)//n}")
    print(f"Decoded matches original: {correct}")

    if not correct:
        num_wrong_bits = sum(a != b for a, b in zip(decoded, test_bits))
        print(f"  -> {num_wrong_bits} bits still wrong after decoding")
