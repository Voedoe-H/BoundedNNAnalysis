import onnx
import onnxruntime as ort
import numpy as np

def nn_montecarlo(input_range,runs,input_layer_size,model):
    """
        Function that does a monte carlo style analysis of the given model. The number of runs defines the number 
        of random inputs that are drawn from the given input_range. Then these are feed forward through the given model
        and finally transformed into intervals of expected outputs 
    """
    random_inputs = np.random.uniform(input_range[0], input_range[1], size=(runs, input_layer_size)).astype(np.float32)
    