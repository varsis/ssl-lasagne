# CREDIT: https://github.com/sbos/np-baselines/blob/master/params.py
# -*- coding: utf-8 -*-
#
# params.py: Implements IO functions for pickling network parameters.
#
import cPickle as pickle
import os

import lasagne as nn

__all__ = [
    'read_model_data',
    'write_model_data',
]

PARAM_EXTENSION = 'params'


def read_model_data(models, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, PARAM_EXTENSION))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    for (model, model_data) in zip(models, data[0:-1]):
        nn.layers.set_all_param_values(model, model_data)
    best_validation_acc = data[-1]
    return best_validation_acc


def write_model_data(models, best_validation_acc, filename):
    """Pickels the parameters within a Lasagne model."""
    data = list(nn.layers.get_all_param_values(model) for model in models) + [best_validation_acc,]
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, PARAM_EXTENSION)
    with open(filename, 'w') as f:
        pickle.dump(data, f)