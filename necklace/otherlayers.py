import numpy as np
import theano.tensor as T
from lasagne.init import GlorotUniform, Constant, Uniform
from lasagne.layers import Layer, DropoutLayer, DenseLayer, MergeLayer
from lasagne.nonlinearities import identity, rectify, softmax


class TransposedDenseLayer(Layer):
    def __init__(self, incoming, num_units, W=GlorotUniform(0.01),
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


class LinearCombinationLayer(MergeLayer):
    def __init__(self, incomings, alpha, **kwargs):
        assert incomings[0].output_shape == incomings[1].output_shape
        super(LinearCombinationLayer, self).__init__(incomings, **kwargs)
        self.alpha = alpha

    def get_output_shape_for(self, input_shapes):
        return self.input_layers[0].get_output_shape_for(input_shapes[0])

    def get_output_for(self, inputs, **kwargs):
        output = (self.alpha[0] * self.input_layers[0].get_output_for(inputs[0], **kwargs) +
                  self.alpha[1] * self.input_layers[1].get_output_for(inputs[1], **kwargs))
        return output


def NecklaceNetwork(incoming, dimensions, LayerClass, tied_weight=False, necklace_link=False, residual_link=False,
                    batch_normalization=False, p_weight=0.5, alpha=0.5):
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
    sparse_layers = []
    feature_layers = []
    last_residual_layer = incoming
    stack_idx = 0
    for _ in range(num_of_stacks):
        stack_str = str(stack_idx)
        stack_idx += 1
        sparse_dimensions = dimensions[_][0:2]
        output_size = dimensions[_][2]
        params_init = [GlorotUniform(0.01),
                       GlorotUniform(0.01),
                       Uniform([0, 0.01])]
        network = LayerClass(network, sparse_dimensions, params_init, [False, None, None, True],
                             name='LISTA_' + stack_str)
        D_list.append(network.get_dictionary_param())
        network = DropoutLayer(network, p=p_weight, name='LISTA_DROP_' + stack_str)
        sparse_layers.append(network)
        network = DenseLayer(network, num_units=output_size, b=None, nonlinearity=identity, name='PROJ_' + stack_str)
        G_list.append(network.W)
        network = DropoutLayer(network, p=p_weight, name='PROJ_DROP_' + stack_str)
        feature_layers.append(network)
        if residual_link:
            network = LinearCombinationLayer(
                [DropoutLayer(DenseLayer(last_residual_layer, num_units=network.output_shape[-1], W=Constant(0.0),
                                         b=None, nonlinearity=identity, name='PROJ_FEA_' + stack_str), p=p_weight),
                 network], [1, 1])
        last_residual_layer = network
    feature = network
    classification_branch = DropoutLayer(DenseLayer(feature, 10, nonlinearity=softmax), p=p_weight)
    for _ in reversed(range(num_of_stacks)):
        stack_str = str(stack_idx)
        stack_idx += 1
        sparse_dimensions = [dimensions[_][0], dimensions[_][1]]
        output_size = num_input if _ == 0 else dimensions[_ - 1][2]
        params_init = [G_list[_] if tied_weight else GlorotUniform(0.01),
                       GlorotUniform(0.01),
                       Uniform([0, 0.01])]
        # make link from encoding feature layer to decoding sparse layer. in case of the inner most feature layer,
        # feature_layer[_] and network are both the inner most layer
        if necklace_link:
            network = LinearCombinationLayer([feature_layers[_], network], [0.5, 0.5])
        network = LayerClass(network, sparse_dimensions, params_init, [tied_weight, None, None, True],
                             name='LISTA_' + stack_str)
        network = DropoutLayer(network, p=p_weight, name='LISTA_DROP_' + stack_str)
        if necklace_link:
            network = LinearCombinationLayer([sparse_layers[_], network], [0.5, 0.5])
        if tied_weight:
            network = TransposedDenseLayer(network, num_units=output_size, W=D_list[_], b=None, nonlinearity=identity,
                                           name='PROJ_' + stack_str)
        else:
            network = DenseLayer(network, num_units=output_size, W=GlorotUniform(0.01), b=None, nonlinearity=identity,
                                 name='PROJ_' + stack_str)
        network = DropoutLayer(network, p=p_weight, name='PROJ_DROP_' + stack_str)
        if residual_link:
            network = LinearCombinationLayer(
                [DropoutLayer(DenseLayer(last_residual_layer, num_units=network.output_shape[-1], W=Constant(0.0),
                                         b=None, nonlinearity=identity, name='PROJ_FEA_' + stack_str), p=p_weight),
                 network], [1, 1])
        last_residual_layer = network
    return network, classification_branch, feature
