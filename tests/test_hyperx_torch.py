import torch
from src.hyperx_torch import HyperXActivation

def test_hyperx_pytorch():
    activation = HyperXActivation(k=1.0)
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    y = activation(x)

    assert y.shape == x.shape
    print("PyTorch Test Passed")

test_hyperx_pytorch()
