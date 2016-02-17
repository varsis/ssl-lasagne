from lasagne.layers import InputLayer, dropout

from otherlayers import NecklaceNetwork
from sparse import LISTAWithDropout


def build_computation_graph(input_var, input_shape, dimensions, p_input=0.2, p_weight=0.5):
    #dimension[-1][-1] is the last output size of last stacked layer a.k.a size of the image vector
    input = InputLayer(shape=input_shape, input_var=input_var, name='input')
    input = dropout(input, p=p_input, name='input_drop')
    network, classification_branch, features = NecklaceNetwork(input, dimensions, LISTAWithDropout, True, True, False,
                                                               False, p_weight)
    return network, classification_branch, features