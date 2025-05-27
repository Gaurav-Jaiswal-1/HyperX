import torch
import tensorflow as tf
from src.hyperx_torch import HyperXActivation
from src.hyperx_tf import HyperXActivation




def test_pytorch_integration():
    class PyTorchModel(torch.nn.Module):
        def __init__(self):
            super(PyTorchModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
            self.hyperx = HyperXActivation(k=1.0)
            self.fc = torch.nn.Linear(32 * 26 * 26, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.hyperx(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = PyTorchModel()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)  # Ensure this works without errors
    assert output.shape == (1, 10)  # Check output shape




class TensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3))
        self.hyperx = HyperXActivation(k=1.0)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.hyperx(x)  # Ensure HyperX handles 4D tensors
        x = self.flatten(x)
        x = self.fc(x)
        print(x.shape)  # After conv1
        return x


import pytest


