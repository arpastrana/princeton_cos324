import numpy as np
from logreg_cd import logloss


rndn = np.random.randn  # normal distribution
rndu = np.random.rand  # uniform distribution, but he doesn't use it


def gimme_data(n, d, m=1):
  # u is a generator
  u = np.arange(d) * (2 * (np.arange(d) % 2) - 1)
  u = u / np.sqrt(np.dot(u, u))  # unitize vector through norm 2

  X = np.random.randint(0, 2, (n, d))  # binary matrix
  X[:,int(d/2)] = 1  # overwrite one of the columns of X 
  # because data num examples >> num dimensions. 
  # Can happen there will be an example where all entries are zero. 
  # Therefore, this is to control the "sparcity" of the simulated data.
  
  z = (m * np.dot(X, u)).reshape(n, 1)  # generate margins
  p = 1 / (1 + np.exp(-z))  # probability

  # Y creates a dataset that is not linearly separable
  # bugs could be everywhere (data, code, algorithm)
  # y helps as a sanity check tossing a probability coin
  y = 2 * (rndu(n, 1) <= p) - 1  

  print('Generator: Loss={0:8.5f}\n'.format(logloss(X, y, u)))  # print stuff

  return X, u, y
