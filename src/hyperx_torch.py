import torch
import torch.nn as nn


class HyperXActivation(nn.Module):
    def __init__(self, k=1.0):
        """
        HyperX Activation Function for PyTorch.
        Args:
            k (float): Trainable parameter controlling the scaling of the input.
        """
        super(HyperXActivation, self).__init__()
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the HyperX activation function.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Transformed tensor after applying the activation function.
        """
        return x * torch.tanh(self.k * x)
