import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from intervals import *
import numpy as np

def gemm_layer(input_intervals, weights, biases):
    """
        Function handling a typical gemm layer
    """
    outputs = []
    for i in range(len(biases)):
        out = Interval(0.0,0.0)
        out = out + biases[i]
        for j in range(len(weights[i])):
            out = out + input_intervals[j] * weights[i][j]
        outputs.append(out)
    return outputs

def convolution_layer():
    # TODO managae convolution layer
    pass

def reshape_layer():
    # TODO manage reshape layer
    pass

def conv_layer(input, weights, biases, kernel_shape, strides, pads, dilations, groups):
    """
    Simplified convolution operation for interval analysis.
    This function should be adapted based on how your 'Interval' class works with convolution.
    """
    # Convert input intervals to a numpy array for easier manipulation if needed
    input_arr = np.array([interval.mid for interval in input]).reshape(*input.shape)
    
    output = []
    
    # Simplified convolution logic; this would need to be adapted for actual interval arithmetic
    for i in range(0, input_arr.shape[0] - kernel_shape[0] + 1, strides[0]):
        for j in range(0, input_arr.shape[1] - kernel_shape[1] + 1, strides[1]):
            # Slice the input for this part of the convolution
            region = input_arr[i:i+kernel_shape[0], j:j+kernel_shape[1]]
            
            # Compute convolution for this region:
            # Note: This is a simplification. Real convolution would involve:
            # - Applying weights with interval arithmetic
            # - Accounting for padding and dilation
            conv_result = Interval(0, 0)  # Start with an empty interval or zero interval
            for ki in range(kernel_shape[0]):
                for kj in range(kernel_shape[1]):
                    for c in range(region.shape[-1]):  # Assuming channel dimension
                        # Use interval arithmetic for multiplication and addition
                        conv_result += input[i+ki, j+kj, c] * Interval(weights[ki, kj, c, :])
            
            conv_result += Interval(biases)  # Add bias if provided
            # Append the result directly to output without reshaping here
            output.append(conv_result)

    # No need to reshape back to any specific tensor shape as we want a flat vector
    return output 



def parseONNXModel(model,inputs):
    """
        Core Interval analysis function that parses through a given onnx model. It does a complete forward pass of the given onnx model
        with the inputs being intervals. Finally returns the ouput intervals of the last layer
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    constants = {} 
    x = inputs
    for node in graph.node:

        # Gemm Layer
        if node.op_type == "Gemm":
            weights_name = node.input[1]
            biases_name = node.input[2] if len(node.input) > 2 else None
            
            weights = onnx.numpy_helper.to_array(initializers[weights_name])
            biases = onnx.numpy_helper.to_array(initializers[biases_name]) if biases_name else [0] * weights.shape[0]   

            x = gemm_layer(x,weights,biases)
        elif node.op_type == "Conv":
            weights_name = node.input[1]
            biases_name = node.input[2] if len(node.input) > 2 else None
            
            weights = onnx.numpy_helper.to_array(initializers[weights_name])
            biases = onnx.numpy_helper.to_array(initializers[biases_name]) if biases_name else np.zeros(weights.shape[0])

            # Extract attributes for convolution
            kernel_shape = [int(attr.i) for attr in node.attribute if attr.name == "kernel_shape"][0]
            strides = [int(attr.i) for attr in node.attribute if attr.name == "strides"][0] if any(attr.name == "strides" for attr in node.attribute) else [1, 1]
            pads = [int(attr.i) for attr in node.attribute if attr.name == "pads"][0] if any(attr.name == "pads" for attr in node.attribute) else [0, 0, 0, 0]
            dilations = [int(attr.i) for attr in node.attribute if attr.name == "dilations"][0] if any(attr.name == "dilations" for attr in node.attribute) else [1, 1]
            groups = [attr.i for attr in node.attribute if attr.name == "group"][0] if any(attr.name == "group" for attr in node.attribute) else 1

            # Assuming 'x' is an Interval object representing an interval tensor
            x = conv_layer(x, weights, biases, kernel_shape, strides, pads, dilations, groups)

        # Relu Layer
        elif node.op_type == "Relu":
            x = [inv.relu() for inv in x ]

        # Sigmoid Layer
        elif node.op_type == "Sigmoid":
            x = [inv.sigmoid() for inv in x]
        
        # Softplus Layer
        elif node.op_type == "Softplus":
            x = [inv.softplus() for inv in x]
        
        # Exponential Layer
        elif node.op_type == "Exp":
            x = [inv.exp() for inv in x]
        
        # Constant Layer
        elif node.op_type == "Constant":
            print("Processing Constant layer")
            for attr in node.attribute:
                if attr.name == "value":
                    # Extract the tensor's value and store it
                    constant_value = onnx.numpy_helper.to_array(attr.t)
                    constants[node.output[0]] = constant_value
        else:
            raise TypeError(f"Non supported lyer typ: {node.op_type}")
        print("-" * 50)    
    return x

def onnx_interval_reduction(onnx_model,activation_function_type):
    """
        Initially expecting just relu activation functions and a single type
        of activation function in the network
    """
    match activation_function_type:
        case ActivationFunction.RELU:
            parse_relu_net(onnx_model)
        case ActivationFunction.TANH:
            parse_tanh_net(onnx_model)
        case _ :
            raise TypeError("Not supported activation funciton type")      


def parse_tanh_net(onnx_model):
    """
        TODO: Write generalized version of this function that handles arbitrary activation function
    """
    graph = onnx_model.graph
    if check_consistency(graph,"Tanh"):
        # As the expected structure is established and the relu nodes are not needed
        # only the relevant weight matricies and bias vectors are gathered 
        # in their sequential occurence in the graph
        weight_matricies = []
        bias_vectors = []
        initializers = {init.name: init for init in graph.initializer}
        for node in graph.node:
            if node.op_type == "Gemm":
                weights_name = node.input[1]
                biases_name = node.input[2] if len(node.input) > 2 else None
                weights = onnx.numpy_helper.to_array(initializers[weights_name])
                biases = onnx.numpy_helper.to_array(initializers[biases_name]) if biases_name else [0] * weights.shape[0]   

                weight_matricies.append(weights)
                bias_vectors.append(biases)
        # Actuall iterate over the weight layers and transform them according to the paper
        # Last layer is treated as output having imaginary weight layer
        
        for j in range(len(weight_matricies)-1):
            A = set() # This is the set that we use the min and max operations on in the paper
            print(weight_matricies[j])
            print(weight_matricies[j+1])
            # Determining the interval that replaces all the incoming edges
            for k in range(len(weight_matricies[j])):
                for p in range(len(weight_matricies[j][k])): 
                    for z in range(len(weight_matricies[j+1])):
                        A.add(weight_matricies[j][k][p]*np.sign(weight_matricies[j+1][z][k]))
            print(A)
            A_array = np.array(list(A))
            interval_min = np.min(A_array)
            interval_max = np.max(A_array)
            print(f"New nodes input edge interval: {interval_min} {interval_max}")
    else:
        raise TypeError("Format of provided RELU onnx model does not match required one")



def parse_relu_net(onnx_model):
    """
        Reduction according to "Interval Weight-Based Abstraction for Neural Network
        Verification" by Boudardara et. al.
        
        In the process the last layer, RELU layer, that goes to output is interpreted as constant weight,
        thus positive 1
    """
    graph = onnx_model.graph
        
    if check_consistency(graph,"Relu"):
        # As the expected structure is established and the relu nodes are not needed
        # only the relevant weight matricies and bias vectors are gathered 
        # in their sequential occurence in the graph
        weight_matricies = []
        bias_vectors = []
        initializers = {init.name: init for init in graph.initializer}
        for node in graph.node:
            if node.op_type == "Gemm":
                weights_name = node.input[1]
                biases_name = node.input[2] if len(node.input) > 2 else None
                weights = onnx.numpy_helper.to_array(initializers[weights_name])
                biases = onnx.numpy_helper.to_array(initializers[biases_name]) if biases_name else [0] * weights.shape[0]   

                weight_matricies.append(weights)
                bias_vectors.append(biases)
        # Actuall iterate over the weight layers and transform them according to the paper
        # Last layer is treated as output having imaginary weight layer
        for j in range(len(weight_matricies)-1):
            pass
        print(weight_matricies)
    else:
        raise TypeError("Format of provided RELU onnx model does not match required one")
    
def check_consistency(onnx_graph,activation_function):
    
    for j in range(len(onnx_graph.node)-1):
        if onnx_graph.node[j].op_type == "Gemm" and not onnx_graph.node[j+1].op_type == activation_function:
            print(onnx_graph.node[j+1].op_type)
            return False
        elif onnx_graph.node[j].op_type == activation_function and not onnx_graph.node[j+1].op_type == "Gemm":
            return False
    return True
