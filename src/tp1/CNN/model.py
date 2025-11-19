import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        input_chanels: int,
        output_channels: int,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_chanels,
                out_channels=output_channels,
                kernel_size=(3, 3),
                padding=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=output_channels),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=output_channels * (input_size // 4),
            hidden_size=hidden_size,
            bidirectional=True,
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x)

        y, _ = self.lstm(x)

        return y
