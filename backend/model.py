import torch
from torch import nn

class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1
        self.layer1 = nn.Linear(in_features=9, out_features=4)

        # ReLU activation
        self.relu = nn.ReLU()

        # Layer 2
        self.layer2 = nn.Linear(in_features=4, out_features=1)

    # forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        return self.layer2(x)