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
    def __init__(self) -> None:
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

    def forward_propagation(self,x):
        """
            x: A numpy array that contains eihter real values or intervals that are either the
            activation from the previous layer or the general inputs inot the network
        """
        for j in range(self.output_size):
            pass