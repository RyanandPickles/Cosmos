import os
import numpy as np
from helpers import bytes_to_lz4_bits, lz4_bits_to_bytes

#dummy byte data
original_bytes = b"Hello SDR! " * 1000  # 11000 bytes of repeating text

print("--- Step 1: Compression ---")
print(f"Original Data Size: {len(original_bytes)} bytes")

#use helper function to compress
tx_bits, compressed_len = bytes_to_lz4_bits(original_bytes)
print(f"Compressed size   : {compressed_len} bytes")
print(f"Total bits output : {len(tx_bits)} bits")
print(
    f"Compression ratio : {(1 - compressed_len / len(original_bytes)) * 100:.2f}% reduced"
)

print("\n--- Step 2: Decompression ---")
#decompress back to bytes
decompressed_bytes = lz4_bits_to_bytes(tx_bits)

#verify integrity
if decompressed_bytes == original_bytes:
    print("Success, decompressed data matches original data")
else:
    print("Faliure, data corrupted during compression/decompression.")
