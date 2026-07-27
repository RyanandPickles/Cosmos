import numpy as np


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
