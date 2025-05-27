import torch
from src.hyperx_torch import HyperXActivation

# Instantiate the activation function
activation = HyperXActivation(k=1.0)

# Input tensor
x = torch.tensor([1.0, -2.0, 3.0, -4.0], requires_grad=True)

# Forward pass
y = activation(x)
print("Output:", y)

# Backward pass (compute gradients)
y.sum().backward()
print("Gradients:", x.grad)
