import torch
from src.custom_layers_torch import HyperXActivation
from torch import nn
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.activation = HyperXActivation(k=1.0)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
