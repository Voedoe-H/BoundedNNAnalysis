from intervals import *
from interval_analysis import *
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

if __name__ == "__main__":
    #example_model_generator()   #Generate test model if needed
    model_path = "small_nn.onnx" # Path to the onnx model, defaulted to the generated small_nn.onnx model
    onnx_model = onnx.load(model_path)

    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")