import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from intervals import *

def gemm_layer(input_intervals, weights, biases):
    """
        Function handling a typical gemm layer
    """
    print(f"weights: {weights}")
    outputs = []
    for i in range(len(biases)):
        out = Interval(0.0,0.0)
        out = out + biases[i]
        print(f"len of weights: {len(weights[i])}")
        for j in range(len(weights[i])):
            print(f"J:{j}")
            print(f"input_intervals[{j}] = {input_intervals[j]} (type: {type(input_intervals[j])})")
            print(f"weights[{i}][{j}] = {weights[i][j]} (type: {type(weights[i][j])})")

            out = out + input_intervals[j] * weights[i][j]
        outputs.append(out)
    return outputs

def convolution_layer():
    # TODO managae convolution layer
    pass

def reshape_layer():
    # TODO manage reshape layer
    pass

def parseONNXModel(model,inputs):
    """
        Core Interval analysis function that parses through a given onnx model. It does a complete forward pass of the given onnx model
        with the inputs being intervals. Finally returns the ouput intervals of the last layer
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    constants = {} 
    x = inputs
    print(x)
    for node in graph.node:
        print(f"Node Name: {node.name}")
        print(f"Operation Type: {node.op_type}")
    
        if node.op_type == "Gemm":
            print("Linear type case")
            weights_name = node.input[1]
            biases_name = node.input[2] if len(node.input) > 2 else None
            
            weights = onnx.numpy_helper.to_array(initializers[weights_name])
            biases = onnx.numpy_helper.to_array(initializers[biases_name]) if biases_name else [0] * weights.shape[0]   

            print(x)
            x = gemm_layer(x,weights,biases)

            print(f"weights: {weights}")
            print(f"biases: {biases}")

        elif node.op_type == "Relu":
            print(x)
            x = [inv.relu() for inv in x ]
            print("Relu type case")

        elif node.op_type == "Constant":
            print("Processing Constant layer")
            for attr in node.attribute:
                if attr.name == "value":
                    # Extract the tensor's value and store it
                    constant_value = onnx.numpy_helper.to_array(attr.t)
                    constants[node.output[0]] = constant_value
                    print(f"Constant value stored: {constant_value}")
        else:
            raise TypeError(f"Non supported lyer typ: {node.op_type}")
        print("-" * 50)    
    return x