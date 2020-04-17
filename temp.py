

import numpy as np
from numpy import array

A = array([[0, 1],  
            [2, 3],  
            [4, 5]])   

B = array([[ 6,  7],  
           [ 8,  9],  
           [10, 11]])

C = A[...,None]*B[:,None,:]

print(C.shape)
print(C)

D= np.einsum('ij,kj->ki', A, B)

print(D.shape)
print(D)

E = np.einsum('ij,kj->ik', A, B)
print(E.shape)
print(E)
