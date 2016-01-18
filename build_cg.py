from lista import LISTALinearStack
from lasagne.layers import InputLayer, dropout

def build_computation_graph(input_var=None, p_input=0.2, p_weight=0.5):
    IM_SIZE = 784
    dimensions = ((1500, 3, 500), (1000, 3, 50),
                  -1,
                  (1000, 3, 500), (1500, 3, IM_SIZE))
    input = InputLayer(shape=(None, IM_SIZE), input_var=input_var, name='input')
    input = dropout(input, p=p_input, name='input_drop')
    network, classification_branch, features = LISTALinearStack(input, dimensions, p_weight)
    return network, classification_branch, features