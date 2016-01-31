from lista import LISTALinearStack
from lasagne.layers import InputLayer, dropout

def build_computation_graph(input_var, dimensions, p_input=0.2, p_weight=0.5):
    #dimension[-1][-1] is the last output size of last stacked layer a.k.a size of the image vector
    input = InputLayer(shape=(None, dimensions[-1][-1]), input_var=input_var, name='input')
    input = dropout(input, p=p_input, name='input_drop')
    network, classification_branch, features = LISTALinearStack(input, dimensions, p_weight)
    return network, classification_branch, features