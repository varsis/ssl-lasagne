import theano.tensor as T
from lasagne.init import GlorotUniform, Normal
from lasagne.layers import Layer, DenseLayer, dropout
from lasagne.nonlinearities import identity

from utils import TransposedDenseLayer


def shrinkage(x, theta):
    return T.switch(T.lt(x, -theta), x + theta, T.switch(T.le(x, theta), 0, x - theta))


class SparseAlgorithm:
    def __init__(self):
        pass

    def get_dictionary_param(self):
        raise NotImplementedError

    def get_dictionary(self):
        raise NotImplementedError


class ShrinkageLayer(Layer):
    def __init__(self, incoming, dict_size, params_init=(GlorotUniform(0.01),
                                                         None,
                                                         Normal(0.0005, mean=0.001)),
                 **kwargs):
        super(ShrinkageLayer, self).__init__(incoming, **kwargs)
        self.dict_size = dict_size
        self.S = self.add_param(params_init[0], [self.dict_size, self.dict_size], name='S',
                                lista=True, lista_weight_W=True, regularizable=True)
        self.B = params_init[1]
        self.theta = self.add_param(params_init[2], [self.dict_size, ], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)

    def get_output_for(self, input, **kwargs):
        eps = 1e-6
        output = shrinkage(T.dot(input, self.S) + self.B.get_output_for(input, **kwargs), self.theta + eps)
        return output

    def get_output_shape_for(self, input_shape):
        return [None, self.dict_size]


class LISTAWithDropout(Layer, SparseAlgorithm):
    def __init__(self, incoming, dimension, params_init=(GlorotUniform(0.01),
                                                         GlorotUniform(0.01),
                                                         Normal(0.0005, mean=0.001)),
                 addition_parameters=[False, None, None], **kwargs):
        '''
        init parameters
        :param incoming: input to the LISTA layer
        :param dimension: 2 numbers list.
         dimension[0] is dict_size, length of dictionary vector in LISTA. dimension[1] is T a.k.a depth
        :param params_init: init value or init method for LISTA
        :transposed: = True if the input dictionary D is the transpose matrix of a theano.compile.SharedVariable V.
         In that case self.W = D^T = V^T^T = V
        :param kwargs: parameters of super class
        :return:
        '''
        super(LISTAWithDropout, self).__init__(incoming, **kwargs)
        num_inputs = incoming.output_shape[-1]
        self.dict_size = dimension[0]
        self.T = dimension[1]
        self.W = self.add_param(params_init[0], [num_inputs, self.dict_size], name='W',
                                lista=True, lista_weight_S=True, sparse_dictionary=True, regularizable=True)
        self.S = self.add_param(params_init[1], [self.dict_size, self.dict_size], name='S',
                                lista=True, lista_weight_W=True, regularizable=True)
        self.theta = self.add_param(params_init[2], [self.dict_size, ], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.transposed = addition_parameters[0]
        self.p_drop_input_to_shrinkage = addition_parameters[1]
        self.p_drop_shrinkage = addition_parameters[2]
        self.input_to_shrinkage = (DenseLayer(incoming, num_units=self.dict_size, W=self.W, b=None,
                                              nonlinearity=identity)
                                   if not self.transposed else
                                   TransposedDenseLayer(incoming, num_units=self.dict_size, W=self.W, b=None,
                                                        nonlinearity=identity))
        if self.p_drop_input_to_shrinkage is not None:
            self.input_to_shrinkage = dropout(self.input_to_shrinkage, self.p_drop_input_to_shrinkage)
        self.output_layer = self.input_to_shrinkage
        for _ in range(self.T):
            self.output_layer = ShrinkageLayer(self.output_layer, self.dict_size,
                                               [self.S, self.input_to_shrinkage, self.theta])
            if self.p_drop_shrinkage is not None and _ is not self.T - 1:
                self.output_layer = dropout(self.output_layer, self.p_drop_shrinkage)

    def get_dictionary_param(self):
        return self.W

    def get_dictionary(self):
        return T.transpose(self.W) if not self.transposed else self.W

    def get_output_for(self, input, **kwargs):
        return self.output_layer.get_output_for(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        return [None, self.dict_size]


class LISTA(Layer, SparseAlgorithm):
    '''
    Class implementation for LISTA transformation
    '''
    def __init__(self, incoming, dimension, params_init=(GlorotUniform(0.01),
                                                         GlorotUniform(0.01),
                                                         Normal(0.0005, mean=0.001)),
                 addition_parameters=[False], **kwargs):
        '''
        init parameters
        :param incoming: input to the LISTA layer
        :param dimension: 2 numbers list.
         dimension[0] is dict_size, length of dictionary vector in LISTA. dimension[1] is T a.k.a depth
        :param params_init: init value or init method for LISTA
        :transposed: = True if the input dictionary D is the transpose matrix of a theano.compile.SharedVariable V.
         In that case self.W = D^T = V^T^T = V
        :param kwargs: parameters of super class
        :return:
        '''
        super(LISTA, self).__init__(incoming, **kwargs)
        num_inputs = incoming.output_shape[-1]
        self.dict_size = dimension[0]
        self.T = dimension[1]
        self.W = self.add_param(params_init[0], [num_inputs, self.dict_size], name='W',
                                lista=True, lista_weight_S=True, sparse_dictionary=True, regularizable=True)
        self.S = self.add_param(params_init[1], [self.dict_size, self.dict_size], name='S',
                                lista=True, lista_weight_W=True, regularizable=True)
        self.theta = self.add_param(params_init[2], [self.dict_size,], name='theta',
                                    lista=True, lista_fun_param=True, regularizable=False)
        self.transposed = addition_parameters[0]

    def get_dictionary_param(self):
        return self.W

    def get_dictionary(self):
        return T.transpose(self.W) if not self.transposed else self.W

    def get_output_for(self, input, **kwargs):
        eps=1e-6
        B = T.dot(input,
                  self.W if not self.transposed else self.W.transpose())
        output = shrinkage(B, self.theta + eps)
        for _ in range(self.T):
            output = shrinkage(T.dot(output, self.S) + B, self.theta + eps)
        return output

    def get_output_shape_for(self, input_shape):
        return [None, self.dict_size]


# -----------------------------OLD FUNCTIONS - MAY NOT WORK NOW----------------------------------------------------------

def SparseLinear(incoming, dimensions, params_init, LayerClass=SparseAlgorithm, p_drop=0.5, stack_index=None):
    '''
    Implementation of a LISTA layer followed by a fully connected layer
    :param incoming: input to the LISTA layer
    :param dict_size: length of dictionary vector in LISTA
    :param depth: number of LISTA iteration
    :param output_size: output size of the fully connected layer
    :param W: init method for W
    :param S: init method for S
    :param theta: init method for theta
    :param p_drop: dropout
    :param stack_index: (optional) stack layer number (used for name clarification in debugging)
    :return:
    '''
    stack_str = str(stack_index) if stack_index is not None else ""
    dimensions = dimensions[0:2]
    output_size = dimensions[2]
    network = LayerClass(incoming, dimensions, params_init, name='LISTA_' + stack_str)
    network = dropout(network, p_drop, name='LISTA_DROP_' + stack_str)
    network = DenseLayer(network, num_units=output_size, b=None, nonlinearity=identity, name='PROJ_' + stack_str)
    network = dropout(network, p=p_drop, name='PROJ_DROP_' + stack_str)
    return network

def SparseLinearStack(incoming, layers_parameters, p_weight=0.5, start_idx=0):
    '''
    Implementation of a stack consists of blocks of Sparse Algo followed by a Linear layer
    :param incoming: input layer
    :param layers_parameters: shape information of each Sparse layer.
     Each element in dimension is a list contains 3 elements: shape information, Sparse Algo Class and parameters
      initialization for the respective block in the stack in respective order.
    :param p_weight: dropout coefficient for weight matrices
    :param SparseLayerType: list of sparse layer type a.k.a LISTA, LCOD, etc
    :return:
    '''
    stack = incoming
    for stack_idx in range(len(layers_parameters)):
        dimensions = layers_parameters[stack_idx][0]
        LayerClass = layers_parameters[stack_idx][1]
        params_init = layers_parameters[stack_idx][2]
        stack = SparseLinear(incoming=stack, dimensions=dimensions, params_init = params_init, LayerClass=LayerClass,
                             p_drop=p_weight, stack_index=start_idx + stack_idx)
    return stack

