import cPickle as pickle

import numpy as np

N_TRAIN = 50000
N_VALID = 10000
N_LABELED = 15000
permute = np.random.permutation(N_TRAIN)
labeled = np.zeros((N_TRAIN,1))
labeled[permute] = 1
with open('../data/labeled_index.pkl', 'w') as f:
    pickle.dump([N_VALID, labeled], f)
