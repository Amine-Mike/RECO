import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        input_chanels: int,
        output_channels: int,
        input_size: int,
        hidden_size: int,
        n_classes: int = 27,
        model_type: str = "LSTM",  # "LSTM", "BI-LSTM" or "GRU"
    ):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_chanels,
                out_channels=output_channels,
                kernel_size=(3, 3),
                padding=1,  # keep same H/W during the convolution so the padding must be at 1
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=output_channels),
            # We put 2 AvgPooling but we could also try only one to reduce the
            # feature map too much and lose information
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 1)),
        )

        # I needed this part to compute the feature size after conv/pooling on height dimension
        # We forward a dummy tensor to determine the output height, this just a
        # dummy going inside my convolutions
        with torch.no_grad():
            dummy_width = 8
            dummy = torch.zeros(1, input_chanels, input_size, dummy_width)
            conv_out = self.conv_layer(dummy)
            _, out_ch, out_h, out_w = conv_out.size()

        self._conv_out_h = out_h
        self._conv_out_w = out_w

        lstm_input_size = out_ch * out_h

        # Bidirectional LSTM
        if model_type == "BI-LSTM":
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                bidirectional=True,
            )
        # Standard LSTM
        elif model_type == "LSTM":
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                bidirectional=False,
            )
        # GRU
        elif model_type == "GRU":
            self.lstm = nn.GRU(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                bidirectional=False,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        hidden_size = hidden_size * 2 if model_type == "BI-LSTM" else hidden_size

        # Final projection from LSTM hidden to classes, the classes are all the
        # letters of the alpahabet with the empty token and a blank space
        self.fc = nn.Linear(hidden_size, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expects x to be either
        - 4D tensor: (batch, channels, height, width)
        - 3D tensor: (channels, height, width) -> batch dim is added
        - 2D tensor: (height, width) -> batch & channel dims are added
        The model returns logits shaped (seq_len, batch, n_classes)
        where seq_len corresponds to the width dimension after convolutions.
        """

        # add batch dimension if needed because the Convolutions need a 4D input
        if x.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)

        if x.dim() != 4:
            raise ValueError(f"CNN expects 4D input (batch, ch, H, W). Got: {x.dim()}D")

        # The conv_layer will output a tensor of shape
        # (Batch_size, out_chanels, out_height, out_width)
        # How I see it:
        #       batch_size is just the size of the batch
        #       out_chanels : Number of output feature maps
        #       out_height : Height of the feature maps after the reduction from the AvgPooling
        #       out_width : Width of the Feature maps ( Basically the lenght of
        #                   the sequence super import for the LSTM )

        x = self.conv_layer(x)  # (batch, out_ch, out_h, out_w)

        # I need some processing before I can go in the LSTM
        # The LSTM needs a 3D tensor but from the conv I get 4D tensor
        # From the dlstp I know that a LSTM needs a tensor of this shape
        # t = (Seq_len, Batch_size, Feature_size)
        # I need to transform the tensor I got from the convolution then
        # I need to permute the out_w is the seq_len
        # Permute to (width, batch, channels, height) and flatten channels*height

        x = x.permute(3, 0, 1, 2)  # (out_w, batch, out_ch, out_h)

        # since I flatten the 3rd dim it will replace it with a dim that will be
        # channels * out_h so basically my feature size
        # I should have a ready tensor to go in the LSTM
        x = x.flatten(2)  # (out_w, batch, ch*out_h) the flatten is primordial

        y, _ = self.lstm(x)  # (seq_len, batch, hidden*2)
        logits = self.fc(y)  # (seq_len, batch, n_classes)

        # Return log-probabilities like MLP so CTCLoss receives log-probs
        return self.log_softmax(logits)
