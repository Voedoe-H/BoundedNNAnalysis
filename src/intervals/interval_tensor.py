
from .interval import Interval

def get_tensor_shape(tensor):
    if isinstance(tensor,list):
        if len(tensor) == 0:
            return [0]
        return [len(tensor)] + get_tensor_shape(tensor[0])
    else:
        return []
    
def get_flatten_tensor(tensor):
    if isinstance(tensor,list):
        return [elem for sublist in tensor for elem in get_flatten_tensor(sublist)]
    else :
        return [tensor]

class IntervalTensor:

    def __init__(self,dta):
        self.data = dta

    def check_elementwise_shape_compatability(self,t2):
        shape1 = self.tensor_dimensions()
        shape2 = t2.tensor_dimensions()
        assert(shape1,shape2)

    def _elementwise_op(self,t1,t2,op):
        """
            Dual elementwise opperation
        """
        if isinstance(t1,list) and isinstance(t2,list):
            return [self._elementwise_op(sub1,sub2,op) for sub1,sub2 in zip(t1,t2)]
        elif isinstance(t1,Interval) and isinstance(t2,Interval):
            return op(t1,t2)
        else:
            raise ValueError("Unsupported")

    def __add__(self,other):
        """
            Basic addition between two IntervalTensors
        """
        # Check if the shapes are equivalent
        self.check_elementwise_shape_compatability(other)
        
        if isinstance(other,IntervalTensor):
            pass
        else:
            raise ValueError("Unsopported")

    def tensor_dimensions(self):
        return get_tensor_shape(self.data)

    def flattened_tensor(self):
        return get_flatten_tensor(self.data)