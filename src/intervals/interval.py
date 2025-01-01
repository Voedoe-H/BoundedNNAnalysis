import numpy as np
import math

# General Helper functions, functions that are used in the interval class but not actually a member of it
def sigmoid(x):
    """
        Basic sigmoid function over the real valued variables x
    """
    return 1/(1+np.exp(-x))

def softplus(x):
    """
        Basic softplus activation function over real valued variables x
    """
    return np.log(1 + np.exp(x))



# Core class
class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f"[{self.lower}, {self.upper}]"

    def __add__(self, other):
        """
            Overloaded Addition
        """
        if isinstance(other,Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        elif isinstance(other,(int,float, np.float32)):
            return Interval(self.lower+other, self.upper + other)
        else:
            raise TypeError(f"Unssupported operant type for +: 'Interval' and '{type(other).__name__}")
    
    def __sub__(self,other):
        """
            Overloaded Subtraction
        """
        if isinstance(other,Interval):
            return Interval(self.lower - other.lower, self.upper - self.lower)
        elif isinstance(other,(int,float, np.float32)):
            return Interval(self.lower-other,self.upper-other)
        else:
            raise TypeError(f"UNsopported operant type for -: 'Interval' and '{type(other).__name__}'")

    def __mul__(self, other):
        """
            Overloaded Multiplication
        """
        if isinstance(other,Interval):
            bounds = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            ]
            return Interval(min(bounds), max(bounds))
        elif isinstance(other,(int,float, np.float32)):
            if other >= 0:
                return Interval(self.lower * other, self.upper * other)
            else:
                return Interval(self.upper * other, self.lower * other)
        else:
            raise TypeError(f"Unsupported operant type for *: {type(other).__name__}")

    def relu(self):
        return Interval(max(0, self.lower), max(0, self.upper))

    def sigmoid(self):
        """
            Sigmoid activation function. Implemented through evaluating lower and upper value with standard real value sigmoid function,
            due to the fact it is montonic.
        """
        return Interval(sigmoid(self.lower),sigmoid(self.upper))

    def tanh(self):
        """
            TODO review
        """
        if self.lower <= 0 and self.upper >= 0:
            return Interval(min(math.tanh(self.lower), 0, math.tanh(self.upper)), max(math.tanh(self.lower), 0, math.tanh(self.upper)))
        else:
            return [min(math.tanh(self.lower), math.tanh(self.upper)), max(math.tanh(self.lower), math.tanh(self.upper))]


    def softplus(self):
        """
            Softplus activation function. Montonic thus simple implementaiton
        """
        return Interval(softplus(self.lower),softplus(self.upper))
    
    def exp(self):
        """
            Exponential activation function
        """
        return Interval(np.exp(self.lower),np.exp(self.upper))