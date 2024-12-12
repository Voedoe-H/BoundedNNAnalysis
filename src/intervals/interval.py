import numpy as np

class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f"[{self.lower}, {self.upper}]"

    def __add__(self, other):
        if isinstance(other,Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        elif isinstance(other,(int,float, np.float32)):
            return Interval(self.lower+other, self.upper + other)
        else:
            raise TypeError(f"Unssupported operant type for +: 'Interval' and '{type(other).__name__}")
    
    def __mul__(self, other):
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
