from intervals import *

import onnx
import onnxruntime as ort
import numpy as np

def nn_montecarlo(input_range,runs,input_layer_size,model_path):
    """
        Function that does a monte carlo style analysis of the given model. The number of runs defines the number 
        of random inputs that are drawn from the given input_range. Then these are feed forward through the given model
        and finally transformed into intervals of expected outputs 
    """

    # Create the onnx session for forward propagation
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Draw the input vectors from the defined input range
    random_inputs = np.random.uniform(input_range[0], input_range[1], size=(runs, input_layer_size)).astype(np.float32) 
    
    # Forward propagation of all the drawn inputs
    results = []
    for j in range(runs):
        input_data = random_inputs[j].reshape(1, -1)
        output = session.run(None, {input_name: input_data})[0]
        results.append(output)
    results = np.array(results)

    # Test output TODO delete later
    mean_output = np.mean(results, axis=0)
    variance_output = np.var(results, axis=0)
    print("Mean Output:", mean_output)
    print("Variance Output:", variance_output)

    # Start of computing the output intervals based on the results of the forward propagations
    mins = np.zeros(input_layer_size)
    maxs = np.zeros(input_layer_size)

    # Set defualt for the minimums and maximus as the first output
    for dim in range(len(results[0][0])):
        mins[dim] = results[0][0][dim]
        maxs[dim] = results[0][0][dim]
    
    for vec in results:
        for dim in range(len(vec[0])):
            if vec[0][dim] < mins[dim]:
                mins[dim] = vec[0][dim]
            elif vec[0][dim] > maxs[dim]:
                maxs[dim] = vec[0][dim]
    
    result_intervals = []
    for j in range(len(mins)):
        print(f"dimension {j}: ({mins[j]}, {maxs[j]})")
        result_intervals.append(Interval(mins[j],maxs[j]))
    
    return result_intervals

def basic_montecarlo(input_range,runs,input_layer_size,model_path):
    """
        Basic Monte Carlo that retusn the just the set of result output vectors
    """
    # Create the onnx session for forward propagation
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Draw the input vectors from the defined input range
    random_inputs = np.random.uniform(input_range[0], input_range[1], size=(runs, input_layer_size)).astype(np.float32) 
    
    # Forward propagation of all the drawn inputs
    results = []
    for j in range(runs):
        input_data = random_inputs[j].reshape(1, -1)
        output = session.run(None, {input_name: input_data})[0]
        results.append(output)
    results = np.array(results)

    return results