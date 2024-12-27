from .interval import Interval
import numpy as np

class IntervalNeuralNetwork:
    def __init__(self) -> None:
        pass

class IntervalLayer:
    def __init__(self,input_size,output_size):
        self.weights = np.array([[Interval[-0.1, 0.1] for _ in range(input_size)] for _ in range(output_size)])
        self.biases = np.array([Interval])