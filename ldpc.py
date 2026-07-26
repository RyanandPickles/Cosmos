# general format for function definition: description, parameters, and returns to stay organized
import numpy as np
# ignores the massive amoutn of 0's in the matrix to speed up
from scipy import sparse

def gen_ldpc_matricies(k,m,column_weight=3,seed=None):
    """
Description:
    Generates the parity check matrix, and other information along with it
Parameters:
    k: # of message bits
    n: # of parity check bits
    column_weight : number of 1s per column of A (numpy array / matrix) --> col_weight check equations
    seed: ensures randomness
Returns:
    A: (m x k) binary numpy array, is a sparse random binary matrix 
    H: (m x n) binary numpy array, parity check matrix
    n: codeword length (m+k)
    """
    #random rng
    rng = np.random.default_rng(seed)
    #initialize m by k matrix
    A = np.zeros((m,k), dtype = np.uint8)
    # loops over columns, assigns 1's randomly
    for column in range(k):
        #prevents more 1's than # of rows available
        weight = min(column_weight, m)
        #no duplicate 1's
        rows = rng.choice(m, size=weight, replace=False)
        A[rows, column] = 1
    
    #identity matrix of size m x m
    I_m = np.eye(m, dtype=np.uint8)

    #creates parity check matrix
    H = np.concatenate((A, I_m), axis=1)
    n = k + m
    return A, H, n

def ldpc_encode_block(message_bits, A):
    """
Description:
    ONE INDIVIDUAL BLOCK - k-bit message --> n-bit codeword
Parameters:
    k-bit message
    A: (m x k) binary numpy array, is a sparse random binary matrix 
Returns:
    codeword: 1D array of 0's and 1's of dimensions 1 x n
    """
    #converts message bits into an array using 8 bit unsigned integer types
    message_bits = np.asarray(message_bits, dtype=np.uint8)
    parity_bits = (A @ message_bits) % 2
    return np.concatenate((message_bits), parity_bits)

def bitstring_to_uint8(bit_string):
    """
Description:
    pretty obvious twin:
Parameters:
    bit string
Returns:
    np.uint8 array
    """
    #Used for storage purposes, ascii takes up one byte with np, otherwise takes more
    return np.frombuffer(bit_string.encode('ascii'), dtype=np.uint8) - ord('0')

def uint8_to_bitstring(bits):
    """
Description:
    pretty obvious twin part 2
Parameters:
    bits in uint8
Returns:
    bit string
    """
    # makes to uint 8, +48 is ord'0', tobytes translates ascii to string, ascii removes b infront of b"string" then 
    return (bits.astype(np.uint8)+ord('0')).tobytes().decode('ascii')

def ldpc_encode(bit_string, A, k):
    """
Description:
    Encode whole bitstring by splitting it into smaller k-length bit strings and encoding them simultaneously
Parameters:
    bitstring
    A matrix: the generator matrix without the identity matrix
    k: length of the individual bit strings
Returns:
    encoded bit string
    original length of the string
    """
    original_length=len(bit_string)

    #In order to get it in multiples of k, add zeros at end of last block
    pad_length = (-original_length) % k
    padded_string = bit_string + '0' * pad_length
    bits = bitstring_to_uint8(padded_string)
    #calculate number of rows needed - e.g. num/k rows
    blocks = bits.reshape(-1,k)




##################################################################################################################################




    #convert to int64 to avoid overflow
    parity = (blocks.astype(np.int64) @ A.T.astype(np.int64)) % 2
    #back down to int8, axis=1 --> horizontal merge
    codewords = np.concatenate((blocks, parity.astype(np.uint8)), axis=1)
    #automatically flattens codewords into a 1-row array and turns it into a bitstring
    encoded_bit_string= uint8_to_bitstring(codewords.reshape(-1))
    return encoded_bit_string, original_length
def ldpc_decode(blocks, H, max_iterations=10, H_sparse=None, Ht_sparse=None):
    """
Description:
    Corrects all of the n-bit blocks recieved at once
Parameters:
    blocks: recieved noisy codewords
    H: parity check matrix
    H_sparse: spicy.sparse CSR version of H, results in much smaller memory being used as arithmetic is ran on only positive 1's
    Ht_sparse: precomputed CSR version of transpose of H
Returns:
    corrected: best guess at codewords
    success: boolean array to see if all parity checks are passed
    """
    if H_sparse is None:
        H_sparse = sparse.csr_matrix(H.astype(np.float32))
    if Ht_sparse is None:
        Ht_sparse = H_sparse.T.tocsr()

    C = blocks.astype(np.float32).copy()
    #reads number of rows
    num_blocks = C.shape[0]
    #looks at which blocks still need fixing
    active = np.ones(num_blocks, dtype=bool)

def ldpc_decode_block(recieved_bits, H, max_iterations=10):
    """
Description:
    Decodes one individual block
Parameters:
    recieved_bits: 1D array of n recieved bits w/ noise
    H: parity check matrix
    max_iterations: max number of iterations to fix otherwise ignore
Returns:
    corrected: 1D array of best guess at true codeword, not necessarily 100% correct
    success: is a boolean check of if all parity checks are passed
    """
    blocks = np.asarray(recieved_bits, dtype=np.uint8).reshape(1,-1)
    corrected, success = ldpc_decode_blocks