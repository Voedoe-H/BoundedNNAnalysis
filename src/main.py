from intervals import *
from interval_analysis import *
from monte_carlo_analysis import *
from visualization import *
import numpy as np
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

    tensor1 = IntervalTensor(
        [Interval(1, 2), Interval(3, 4)]
        )
    tensor2 = IntervalTensor([Interval(5, 6), Interval(7, 8)])
    res_add = tensor1 + tensor2
    res_sub = tensor1 - tensor2
    res_mul = tensor1 * tensor2
    print(res_add.data)
    print(res_sub.data)
    print(res_mul.data)

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

def example_model_generator_tanh():
    """
    Function generating a very small NN with torch using tanh activations to test
    """
    class SmallNN(nn.Module):
        def __init__(self):
            super(SmallNN, self).__init__()
            self.fc1 = nn.Linear(4, 3)  # Input size 4, Output size 3
            self.tanh1 = nn.Tanh()
            self.fc2 = nn.Linear(3, 2)  # Input size 3, Output size 2
            self.tanh2 = nn.Tanh()

        def forward(self, x):
            x = self.fc1(x)
            x = self.tanh1(x)
            x = self.fc2(x)
            x = self.tanh2(x)
            return x

    model = SmallNN()
    dummy_input = torch.randn(1, 4)  # Batch size 1, Input size 4
    onnx_file = "small_nn_tanh.onnx"
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

def convolution_layer_testing():
    # Define a simple neural network with just a convolution layer
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super(SimpleConvNet, self).__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            return self.conv(x)

    # Initialize the model
    model = SimpleConvNet()

    # Create a dummy input tensor for ONNX export
    dummy_input = torch.randn(1, 1, 5, 5)  # batch_size, channels, height, width

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, "simple_conv.onnx", export_params=True, opset_version=11)

    # Load the ONNX model
    onnx_model = onnx.load("simple_conv.onnx")

    # Test with constant interval inputs
    constant_value = 1.0  # or any other constant value
    interval_input = [Interval(constant_value,constant_value) for _ in range(25)]  # 5x5 input, each element represented by an interval

    # Run the interval analysis
    interval_output = parseONNXModel(onnx_model, interval_input)

    # Now let's run the model with ONNX Runtime for comparison
    ort_session = ort.InferenceSession("simple_conv.onnx")

    # Prepare input for ONNX Runtime (convert to numpy)
    ort_input = np.full((1, 1, 5, 5), constant_value, dtype=np.float32)

    # Run the model
    ort_output = ort_session.run(None, {"input": ort_input})[0]

    # Compare results
    print("Interval Analysis Output:")
    for interval in interval_output:
        print(interval)

    print("\nONNX Runtime Output:")
    print(ort_output.flatten())

    # Check if the outputs match
    for i, (interval, ort_val) in enumerate(zip(interval_output, ort_output.flatten())):
        assert np.isclose(interval.lower, ort_val), f"Mismatch at index {i}: Interval {interval.lower} vs ONNX {ort_val}"
    print("\nAll values match within tolerance!")

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

def basic_monte_carlo(model_path):
    """
    """
    input_range = (0,1)
    res = basic_montecarlo(input_range,10000,785,model_path)

    twoDRes = []
    for vec in res:
        twoDRes.append(vec[0])
    scatter_montecarlo(twoDRes)

def interval_reduction():
    #net = IntervalNeuralNetwork(input_size=2, output_size=1)
    #net.add_layer(IntervalLayer(2,2,activation_function = ActivationFunction.RELU))
    #net.add_layer(IntervalLayer(2,1,activation_function = ActivationFunction.SIGMOID))
    #input = [Interval(0.0,0.0),Interval(0.0,0.0)]
    #output = net.forward(input)
    #for out in output:
    #    print(out)
    model_path = "small_nn_tanh.onnx"
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    onnx_interval_reduction(onnx_model,ActivationFunction.TANH)

if __name__ == "__main__":
    #example_model_generator()   #Generate test model if neede
    #minst_path = "mnist_mlp_noreshape.onnx"
    #model_path = "small_nn.onnx" # Path to the onnx model, defaulted to the generated small_nn.onnx model
    #onnx_model = onnx.load(model_path)
    #mnist_model = onnx.load(minst_path)
    #onnx.checker.check_model(onnx_model)
    #onnx.checker.check_model(mnist_model)
    #print("ONNX model is valid!")

    # Example interval analysis replace with your code here
    #interval_results = interval_analysis_example(onnx_model)

    # Example monte carlo analysis replace with your code here
    #monte_carlo_results = monte_carlo_analysis_example(model_path)

    # Some simple visualizaiton option example
    #interval_set_comparison(interval_results[:2],monte_carlo_results[:2],"interval analysis","monte carlo analysis","Output Intervals: Interval Analysis vs Monte Carlo")
    #basic_monte_carlo(mnist_model)
    #constant_layer()
    #tensor_testing()

    #convolution_layer_testing()
    
    #example_model_generator_tanh()
    interval_reduction()
    #example_model_generator()