import time
import numpy as np
import theano
import matplotlib.pyplot as plt
import params_io as io
import cPickle as pickle
from lasagne import updates, layers, objectives, regularization, utils
from theano import tensor as T
from build_cg import build_computation_graph
from load import mnist

def iterate_minibatches(inputs, targets, labeled, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], labeled[excerpt]

def repeat_col(col, n_col):
    return np.repeat(col, n_col, axis=1)

#-----------------------PARAMETERS----------------------------#

with open('labeled_index.pkl', 'r') as f:
    loaded_obj = pickle.load(f)
VALIDATION_SIZE = loaded_obj[0]
labeled_idx = loaded_obj[1]

print('Loading data')

trX, vlX, teX, trY, vlY, teY = mnist(onehot=True, ndim=2)
_, _, _, lbY, _ , _ = mnist(onehot=False)

print('Building computation graph')

IM_SIZE = trX.shape[1]
# # dimensions = ((1500, 3, 500), (1000, 3, 100),
# #               -1,
# #               (1000, 3, 500), (1500, 3, IM_SIZE))
# dimensions = ((1500, 3, 100),
#               -1,
#               (1500, 3, IM_SIZE))
dimensions = [[1500, 3, 100]]
input_shape = [None, IM_SIZE]
input_var = T.fmatrix('input_var')
target_var = T.fmatrix('target_var')
labeled_var = T.fmatrix('labeled_var')
unsupervised_graph, supervised_graph, features = build_computation_graph(input_var, input_shape, dimensions)

lr = (1.0, 1, 1e-2)

reconstruction = layers.get_output(unsupervised_graph)
prediction = layers.get_output(supervised_graph)
params = layers.get_all_params(unsupervised_graph, trainable=True) + \
         layers.get_all_params(supervised_graph, trainable=True)
params = utils.unique(params)

regularization_params = layers.get_all_params(unsupervised_graph, regularizable=True) + \
         layers.get_all_params(supervised_graph, regularizable=True)
regularization_params = utils.unique(regularization_params)
regularization_params = [regularization_params.pop()]

loss1 = objectives.squared_error(reconstruction, input_var)
loss2 = objectives.squared_error(prediction, target_var)
l2_penalties = regularization.apply_penalty(regularization_params, regularization.l2)
loss = lr[0]*loss1.mean() +\
       lr[1]*(loss2*repeat_col(labeled_var, 10)).mean() +\
       lr[2]*l2_penalties.mean()

updates = updates.adam(loss,params,0.00001)
test_reconstruction = layers.get_output(unsupervised_graph, deterministic=True)
test_prediction = layers.get_output(supervised_graph, deterministic=True)
test_loss1 = objectives.squared_error(test_reconstruction, input_var)
test_loss2 = objectives.squared_error(test_prediction, target_var)
test_loss = lr[0]*test_loss1.mean() +\
            lr[1]*test_loss2.mean() +\
            lr[2]*l2_penalties.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var, labeled_var], loss, updates=updates, allow_input_downcast=True,
                           on_unused_input='ignore')
# Compile a second function computing the validation loss and accuracy:
#val_fn = theano.function([input_var, target_var, labeled_var], [loss2*lr[1], test_acc], allow_input_downcast=True)
val_fn = theano.function([input_var, target_var, labeled_var],
                         [test_loss,
                          lr[0]*test_loss1.mean(),
                          lr[1]*test_loss2.mean(),
                          lr[2]*l2_penalties.mean(),
                          test_acc], allow_input_downcast=True,
                         on_unused_input='ignore')

print 'Training...'
num_epochs=3000
train_loss = []
train_loss1 = []
train_loss2 = []
train_regularize = []
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in iterate_minibatches(trX, trY, labeled_idx, 500, shuffle=True):
        inputs, targets, labeled= batch
        train_err = train_fn(inputs, targets, labeled)

    train_err = 0
    train_acc = 0
    train_batches = 0
    train_loss.append(0)
    train_loss1.append(0)
    train_loss2.append(0)
    train_regularize.append(0)
    for batch in iterate_minibatches(trX, trY, labeled_idx, 500, shuffle=True):
        inputs, targets, labeled = batch
        err, _loss1, _loss2, _regularize, acc = val_fn(inputs, targets, labeled)
        train_loss[-1] += err
        train_loss1[-1] += _loss1
        train_loss2[-1] += _loss2
        train_regularize[-1] += _regularize
        train_err += err
        train_acc += acc
        train_batches += 1
    train_loss[-1] /= train_batches
    train_loss1[-1] /= train_batches
    train_loss2[-1] /= train_batches
    train_regularize[-1] /= train_batches

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(vlX, vlY,
                                     np.zeros((vlY.shape[0], 1)),
                                     500, shuffle=False):
        inputs, targets, labeled = batch
        err, _, _, _, acc = val_fn(inputs, targets, labeled)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.6f} %".format(
        train_acc / train_batches * 100))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(teX, teY,
                                     np.zeros((teY.shape[0], 1)),
                                     500, shuffle=False):
        inputs, targets, _ = batch
        err, _, _, _, acc = val_fn(inputs, targets, labeled)
        test_err += err
        test_acc += acc
        test_batches += 1
    #print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

plt.clf()
plt.plot(train_loss,'r-')
plt.plot(train_loss1,'g-')
plt.plot(train_loss2,'b-')
plt.plot(train_regularize,'k-')
plt.show()
