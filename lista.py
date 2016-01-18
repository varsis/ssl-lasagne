import theano.tensor as T
from lasagne.init import GlorotUniform, Normal
from lasagne.layers import Layer, InputLayer, DenseLayer, dropout
from lasagne.nonlinearities import identity, softmax

def shrinkage(x, theta):
    return T.switch(T.lt(x, -theta), x + theta, T.switch(T.le(x, theta), 0, x - theta))


class LISTA(Layer):
    '''
    Class implementation for LISTA transformation
    '''
    def __init__(self, incoming, dict_size, depth,
                 W=GlorotUniform(0.01),
                 S=GlorotUniform(0.01),
                 theta=Normal(0.0005, mean=0.001),
                 **kwargs):
        '''
        init
        :param incoming: input to the LISTA layer
        :param dict_size: length of dictionary vector in LISTA
        :param depth: number of LISTA iteration
        :param W: init method for W
        :param S: init method for S
        :param theta: init method for theta
        :param kwargs: parameters of super class
        :return:
        '''
        super(LISTA, self).__init__(incoming, **kwargs)
        num_inputs = incoming.output_shape[1]
        self.dict_size = dict_size
        self.W = self.add_param(W, (num_inputs, dict_size), name='W',
                                lista=True, role='weight', regularizable=False)
        self.S = self.add_param(S, (dict_size, dict_size), name='S',
                                lista=True, role='bias', regularizable=True)
        self.theta = self.add_param(theta, (dict_size,), name='theta',
                                    lista=True, role='bias', regularizable=False)
        self.T = depth

    def get_output_for(self, input, **kwargs):
        B = T.dot(input, self.W)
        output = shrinkage(B, self.theta)
        for _ in range(0, self.T):
            output = shrinkage(T.dot(output, self.S) + B, self.theta)
        return output

    def get_output_shape_for(self, input_shape):
        return (None, self.dict_size)

def LISTALinear(incoming, dict_size, depth, output_size,
                W=GlorotUniform(0.01),
                S=GlorotUniform(0.01),
                theta=Normal(0.0005, mean=0.001),
                p_drop=0.5, stack_index=0):
    '''
    implementation of a LISTA layer followed by a fully connected layer
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
    network = LISTA(incoming, dict_size, depth, W, S, theta, name='LISTA_' + str(stack_index))
    network = dropout(network, p_drop, name='LISTA_DROP_' + str(stack_index))
    network = DenseLayer(network, num_units=output_size, nonlinearity=identity,
                         name='PROJ_' + str(stack_index))
    network = dropout(network, p=p_drop, name='PROJ_DROP_' + str(stack_index))
    return network

def LISTALinearStack(incoming, dimensions, p_weight=0.5):
    '''
    implementation of stacked LISTA layers
    :param incoming: input to the first LISTA layer
    :param dimensions: shape information of a LISTA layer. If dimension[idx]==-1: create the classification branch.
    Else dimensions[idx] is a list with 3 number implying dict_size, depth, output_size for LISTALinear
    :param p_weight: dropout coefficient for weight matrices
    :return:
    '''
    network = incoming
    for stack_idx in range(len(dimensions)):
        if dimensions[stack_idx] == -1:
            classification_branch = dropout(DenseLayer(network, 10, nonlinearity=softmax),name='CLASSIFY')
            classification_branch = dropout(classification_branch, p_weight)
            features = network
        else:
            network = LISTALinear(network,
                                  dimensions[stack_idx][0],
                                  dimensions[stack_idx][1],
                                  dimensions[stack_idx][2],
                                  stack_index=stack_idx)
    return network, classification_branch, features
