import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperXActivation(nn.Module):
    """
    PyTorch implementation of the HyperX Activation Function: x * tanh(kx)
    """
    def __init__(self, k=1.0):
        super(HyperXActivation, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Forward pass of the activation function.
        
        Parameters:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output of the activation function.
        """
        return x * torch.tanh(self.k * x)
