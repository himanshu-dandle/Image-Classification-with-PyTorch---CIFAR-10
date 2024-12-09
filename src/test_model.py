import torch
from train import SimpleCNN, prepare_data

def test_data_loading():
    trainloader, testloader = prepare_data()
    inputs, labels = next(iter(trainloader))
    assert inputs.shape == (32, 3, 32, 32), "Trainloader input shape mismatch!"
    assert labels.shape == (32,), "Trainloader label shape mismatch!"

def test_model_output():
    model = SimpleCNN()
    inputs = torch.randn(32, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (32, 10), "Model output shape mismatch!"

test_data_loading()
test_model_output()
print("All tests passed!")
