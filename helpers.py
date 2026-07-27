import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import random

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