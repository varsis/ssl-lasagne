import cPickle as pickle
import time
import numpy as np
import theano
from lasagne import updates, layers, objectives, regularization
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

def remove_duplicate(list):
    out = []
    [out.append(elem) for elem in list if elem not in out]
    return out

with open('labeled_index.pkl', 'r') as f:
    loaded_obj = pickle.load(f)
VALIDATION_SIZE = loaded_obj[0]
labeled_idx = loaded_obj[1]

print('Loading data')

trX, vlX, teX, trY, vlY, teY = mnist(onehot=True, ndim=2)
_, _, _, lbY, _ , _ = mnist(onehot=False)

print('Building computation graph')

input_var = T.fmatrix('input_var')
target_var = T.fmatrix('target_var')
labeled_var = T.fmatrix('labeled_var')
unsupervised_graph, supervised_graph, features = build_computation_graph(input_var)

lr = (1.0, 1.0, 0)

reconstruction = layers.get_output(unsupervised_graph)
prediction = layers.get_output(supervised_graph)
params = layers.get_all_params(unsupervised_graph, trainable=True) + \
         layers.get_all_params(supervised_graph, trainable=True)
params = remove_duplicate(params)

loss1 = objectives.squared_error(reconstruction, input_var).mean()
loss2 = (objectives.squared_error(prediction, target_var) * labeled_var).mean()
regularization_params = layers.get_all_params(unsupervised_graph, regularizable=True) + \
         layers.get_all_params(supervised_graph, regularizable=True)
regularization_params = remove_duplicate(regularization_params)
l2_penalties = regularization.apply_penalty(params, regularization.l2)
loss = loss1*lr[0] + loss2*lr[1] + l2_penalties*lr[2]

updates = updates.adam(loss,params,0.00001)
test_prediction = layers.get_output(supervised_graph, deterministic=True)
test_loss = objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var, labeled_var], loss, updates=updates, allow_input_downcast=True,
                           on_unused_input='ignore')
# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var, labeled_var], [loss2*lr[1], test_acc], allow_input_downcast=True)

print 'Training...'
num_epochs=500
for epoch in range(num_epochs):
        start_time = time.time()
        for batch in iterate_minibatches(trX, trY, repeat_col(labeled_idx, 10), 500, shuffle=True):
            inputs, targets, labeled= batch
            train_err = train_fn(inputs, targets, labeled)

        train_err = 0
        train_acc = 0
        train_batches = 0
        for batch in iterate_minibatches(trX, trY, repeat_col(labeled_idx, 10), 500, shuffle=True):
            inputs, targets, labeled = batch
            err, acc = val_fn(inputs, targets, labeled)
            train_err += err
            train_acc += acc
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(vlX, vlY,
                                         repeat_col(np.asarray(range(vlY.shape[0])).reshape(vlY.shape[0], 1), 10),
                                         500, shuffle=False):
            inputs, targets, labeled = batch
            err, acc = val_fn(inputs, targets, labeled)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.6f}".format(train_acc / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(teX, teY,
                                         repeat_col(np.asarray(range(teY.shape[0])).reshape(teY.shape[0], 1), 10),
                                         500, shuffle=False):
            inputs, targets, _ = batch
            err, acc = val_fn(inputs, targets, labeled)
            test_err += err
            test_acc += acc
            test_batches += 1
        #print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))