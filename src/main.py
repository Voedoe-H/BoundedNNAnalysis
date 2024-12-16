from intervals import *
from interval_analysis import *
from monte_carlo_analysis import *
from visualization import *

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import helper, TensorProto

def constant_layer():
    """
        Playing with constant layer
    """

    constant_tensor = onnx.helper.make_tensor(
        name="constant_value",
        data_type=TensorProto.FLOAT,
        dims=[4],  # Flattened shape
        vals=[1.0, 2.0, 3.0, 4.0],  # Values
    )

    # Define the Constant node
    constant_node = onnx.helper.make_node(
        "Constant",
        inputs=[],  # No inputs
        outputs=["output_tensor"],
        value=constant_tensor,  # Constant tensor
    )

    # Define the graph
    graph = helper.make_graph(
        nodes=[constant_node],
        name="ConstantGraph",
        inputs=[],  # No inputs required for this model
        outputs=[
            helper.make_tensor_value_info(
                name="output_tensor",
                elem_type=TensorProto.FLOAT,
                shape=[4],  # Flattened output shape
            )
        ],
    )

    # Define the model
    model = helper.make_model(graph, producer_name="onnx-example")

    parseONNXModel(model,[])
    
def tensor_testing():
    """
        Playing Interval Tensors
    """
    tensor = [
        [  # Channel 1
            [Interval(1, 2), Interval(3, 4)],  # Row 1
            [Interval(5, 6), Interval(7, 8)]   # Row 2
        ],
        [  # Channel 2
            [Interval(9, 10), Interval(11, 12)],  # Row 1
            [Interval(13, 14), Interval(15, 16)]  # Row 2
        ]
    ]
    intervalTensor = IntervalTensor(tensor)
    print(intervalTensor.tensor_dimensions())
    print(intervalTensor.flattened_tensor())

def example_model_generator():
    """
        Function generating a very small NN with torch to test
    """
    class SmallNN(nn.Module):
        def __init__(self):
            super(SmallNN, self).__init__()
            self.fc1 = nn.Linear(4, 3)  # Input size 4, Output size 3
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(3, 2)  # Input size 3, Output size 2
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            return x
    model = SmallNN()
    dummy_input = torch.randn(1, 4)  # Batch size 1, Input size 4
    onnx_file = "small_nn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def interval_analysis_example(model):
    """
        Example function utilizing the small_nn model
    """
    # First off we need to define what inputs we want to analyse:
    inputs =  [Interval(0.0, 1.0) for _ in range(4)] # Input regions are dependend on what scenarios need to be looked at
    res = parseONNXModel(model,inputs) # Actual forward passing the intervals through the NN with res being a list of interval objects
    print("-"*50)
    print("Interval Analysis Results:")
    for j in range(len(res)):
        print(f"Output interval for neuron {j}: ({res[j].lower}, {res[j].upper})")

    return res

def monte_carlo_analysis_example(model_path):
    """
        Example function for the monte carlo analysis based on the small_nn model
    """
    input_range = (0, 1)
    res = nn_montecarlo(input_range,100000,4,model_path)
    print("-"*50)
    print("Monte Carlo Analysis Results:")
    for j in range(len(res)):
        print(f"Output interval for neuron {j}: ({res[j].lower}, {res[j].upper})")

    return res

if __name__ == "__main__":
    #example_model_generator()   #Generate test model if needed
    model_path = "small_nn.onnx" # Path to the onnx model, defaulted to the generated small_nn.onnx model
    onnx_model = onnx.load(model_path)

    #onnx.checker.check_model(onnx_model)
    #print("ONNX model is valid!")

    # Example interval analysis replace with your code here
    interval_results = interval_analysis_example(onnx_model)

    # Example monte carlo analysis replace with your code here
    monte_carlo_results = monte_carlo_analysis_example(model_path)

    # Some simple visualizaiton option example
    interval_set_comparison(interval_results[:2],monte_carlo_results[:2],"interval analysis","monte carlo analysis")
    
    #constant_layer()
    #tensor_testing()