import numpy as np

A = np.array([0, 1, 2])

B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

A = A.reshape(-1, 1)

# C = (A[:, np.newaxis] * B).sum(axis=1)
C = (A * B).sum(axis=1)
print('C', C)

D = np.einsum('ij,ij->i', A, B)
print('D', D)
print(D.shape)
