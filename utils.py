import theano.tensor as T
import numpy as np
from lasagne.init import GlorotUniform, Constant, Normal
from lasagne.layers import Layer, dropout, DenseLayer
from lasagne.nonlinearities import identity, rectify, softmax
from sparse import SparseAlgorithm


class TransposedDenseLayer(Layer):

    def __init__(self, incoming, num_units, W=GlorotUniform(),
                 b=Constant(0.), nonlinearity=rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W.transpose())
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class LinearCombinationLayer(Layer):
    def __init__(self, incoming1, incoming2, alpha=0.5, **kwargs):
        assert incoming1.output_shape == incoming2.output_shape
        super(LinearCombinationLayer, self).__init__(incoming1, **kwargs)
        self.incoming1 = incoming1
        self.incoming2 = incoming2
        self.alpha = alpha


def NecklaceNetwork(incoming, dimensions, LayerClass=SparseAlgorithm, p_weight=0.5, alpha=0.5):
    '''
    Implementation of a necklace network. See: https://drive.google.com/file/d/0B8EOfHp2L5mNbkY1UkhlWmF2YWc/view?ts=56b3a4fc
    :param dimensions: contain information of the dimension
    :param LayerClass: Sparse Code Algorithm class
    :param p_weight: dropout factor for weight matrices
    :param alpha: alpha factor in x' = alpha*a + (1-alpha)*a'
    :return:
    '''
    network = incoming
    num_input = incoming.output_shape[1]
    num_of_stacks = len(dimensions)
    D_list = []
    G_list = []
    stack_idx=0
    for _ in range(num_of_stacks):
        stack_str = str(stack_idx)
        stack_idx += 1
        sparse_dimensions = dimensions[_][0:2]
        output_size = dimensions[_][2]
        params_init=[GlorotUniform(0.01),
             GlorotUniform(0.01),
             Normal(0.0005, mean=0.001)]
        network = LayerClass(incoming, sparse_dimensions, params_init, transposed=False, name='LISTA_' + stack_str)
        D_list.append(network.get_dictionary_param())
        network = dropout(network, p_weight, name='LISTA_DROP_' + stack_str)
        network = DenseLayer(network, num_units=output_size, b=None, nonlinearity=identity, name='PROJ_' + stack_str)
        G_list.append(network.W)
        network = dropout(network, p=p_weight, name='PROJ_DROP_' + stack_str)
    feature = network
    classification_branch = dropout(DenseLayer(feature, 10, nonlinearity=softmax), p_weight)
    for _ in reversed(range(num_of_stacks)):
        stack_str = str(stack_idx)
        stack_idx += 1
        sparse_dimensions = [dimensions[_][0], dimensions[_][1]]
        output_size = num_input if _ == 0 else dimensions[_-1][0]
        params_init = [G_list[_],
                       GlorotUniform(0.01),
                       Normal(0.0005, mean=0.001)]
        network = LayerClass(network, sparse_dimensions, params_init, transposed=True, name='LISTA_' + stack_str)
        network = dropout(network, p_weight, name='LISTA_DROP_' + stack_str)
        network = TransposedDenseLayer(network, num_units=output_size, W=D_list[_], b=None, nonlinearity=identity, name='PROJ_' + stack_str)
        network = dropout(network, p=p_weight, name='PROJ_DROP_' + stack_str)
    return network, classification_branch, feature
