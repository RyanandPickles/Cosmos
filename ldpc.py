import numpy as np

def gen_ldpc_matricies(k,m,col_weight=3,seed=None)
    """
Parameters:
    k: # of message bits
    n: # of parity check bits
    col_weight : number of 1s per column of A (numpy array / matrix) --> col_weight check equations
    seed: ensures randomness
Returns:
    A: (m x k) binary numpy array
    H: (m x n) binary numpy array, parity check matrix
    n: codeword length (m+k)
    """

    A = np.zeros((m,k), dtype = np.uint8)

