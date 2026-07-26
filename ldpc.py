import numpy as np

def gen_ldpc_matricies(k,m,column_weight=3,seed=None)
    
    """
Parameters:
    k: # of message bits
    n: # of parity check bits
    column_weight : number of 1s per column of A (numpy array / matrix) --> col_weight check equations
    seed: ensures randomness
Returns:
    A: (m x k) binary numpy array
    H: (m x n) binary numpy array, parity check matrix
    n: codeword length (m+k)
    """
    #random rng
    rng = np.random.default_rng(seed)
    #initialize m by k matrix
    A = np.zeros((m,k), dtype = np.uint8)
    # loops over columns, assigns 1's randomly
    for columns in range(k):
        #prevents more 1's than # of rows available
        weight = min(column_weight, m)
        #no duplicate 1's
        rows = rng.choice(m, size=weight, replace=False)
        A[rows, col] = 1
    
    #identity matrix of size m x m
    I_m = np.eye(m, dtype=np.uint8)

    #creates parity check matrix
    H = np.concatenate((A, I_m), axis=1)
    n = k + m
    return A, H, n
