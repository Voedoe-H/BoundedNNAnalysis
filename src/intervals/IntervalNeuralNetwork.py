from .interval import Interval
import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    """
        Enum class for all the different activation functions for the forward propagation
    """
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    LINEAR = 'linear'
    TANH = 'tanh'

class IntervalNeuralNetwork:
    def __init__(self,input_size,output_size) -> None:
        """
        
        """
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden = 0

    def add_layer(self,layer):
        """
        
        """
        self.layers.append(layer)
        self.num_hidden +=1
    
    def forward(self,input):
        """
        
        """
        if len(input) != self.input_size:
            raise ValueError("Input size does not match with given input")
        else:
            pass

class IntervalLayer:

    def __init__(self,input_size,output_size,activation_function):
        """
            input_size: Integer defining the number of neurons in the previous layer
            output_size: Integer defining the number of neurons in this layer
            activation_function: enum that defines which specific activation function should be used for the forward propagation
        """
        self.weights = np.array([[Interval(-0.1, 0.1) for _ in range(input_size)] for _ in range(output_size)])
        self.biases = np.array([Interval(0.0,1.0) for _ in range(output_size)])
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def forward_propagation(self,inputs):
        """
            inputs: A numpy array that contains eihter real values or intervals that are either the
            activation from the previous layer or the general inputs inot the network
        """
        outputs = np.array([])
        for j in range(self.output_size):
            z = self.weight_mul(inputs,self.weights[j])
            outputs.append(self.apply_activation(z))
        
        return outputs
    
    def weight_mul(self,inputs,weights):
        """
        
        """
        res = Interval(0.0,0.0)
        for w,x in zip(weights,inputs):
            res + (w*x)
        return res
    
    def apply_activation(self,z):
        """
        
        """
        match self.activation_function:
            case ActivationFunction.RELU:
                return z.relu()
            case ActivationFunction.SIGMOID:    
                return  z.sigmoid()
            case ActivationFunction.LINEAR:
                return(z)
            case ActivationFunction.TANH:
                # TODO : implement tanh for intervals. maybe bounded 
                return(z)
            case _ :
                raise TypeError("Unidentified activation function")
       