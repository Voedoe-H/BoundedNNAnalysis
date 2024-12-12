from intervals import *
from interval_analysis import *
from monte_carlo_analysis import *

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

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
    for j in range(len(res)):
        print(f"Output interval for neuron {j}: ({res[j].lower}, {res[j].upper})")

if __name__ == "__main__":
    #example_model_generator()   #Generate test model if needed
    model_path = "small_nn.onnx" # Path to the onnx model, defaulted to the generated small_nn.onnx model
    onnx_model = onnx.load(model_path)

    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Example interval analysis replace with your code here
    interval_analysis_example(onnx_model)
