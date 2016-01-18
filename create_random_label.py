import cPickle as pickle

import numpy as np

VALIDATION_SIZE = 10000
NUM_LABELED = 15000
permute = np.random.permutation(60000 - VALIDATION_SIZE)
labeled = np.zeros((60000,1))
labeled[permute[1:NUM_LABELED]] = 1
with open('labeled_index.pkl', 'w') as f:
    pickle.dump([VALIDATION_SIZE, labeled], f)
