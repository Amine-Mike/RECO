import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, n_layers: int
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * n_layers),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        return self.layers(x)
