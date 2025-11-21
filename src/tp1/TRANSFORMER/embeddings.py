import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.s_emb = nn.Embedding(num_embeddings=num_vocab, embedding_dim=num_hid)
        self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=num_hid)

    def forward(self, x):
        maxlen = x.size(1)
        x = self.s_emb(x)
        positions = torch.arange(maxlen, device=x.device)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(nn.Module):
    """
    Process raw spectrogram features using convolution layers and positional encoding.
    Based on the original Keras implementation that processes STFT spectrograms.
    """

    def __init__(self, num_hid=64, maxlen=2754):
        """
        Args:
            num_hid: Hidden dimension (output embedding size)
            maxlen: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.num_hid = num_hid

        # Convolutional layers to process spectrogram features
        # Input: [batch, time, freq=129] from STFT
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=129,
                out_channels=num_hid,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_hid,
                out_channels=num_hid,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_hid,
                out_channels=num_hid,
                kernel_size=11,
                stride=2,
                padding=5,
            ),
            nn.ReLU(),
        )

        # Positional encoding
        # After 3 conv layers with stride 2, maxlen becomes maxlen // 8
        pos_maxlen = maxlen // 8 + 1
        self.pos_emb = nn.Embedding(num_embeddings=pos_maxlen, embedding_dim=num_hid)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram [batch_size, time, freq]

        Returns:
            Embedded features [batch_size, time//8, num_hid]
        """
        # Transpose for Conv1d: [batch, time, freq] -> [batch, freq, time]
        x = x.transpose(1, 2)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Transpose back: [batch, num_hid, time] -> [batch, time, num_hid]
        x = x.transpose(1, 2)

        # Add positional encoding
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_emb(positions)

        return x + pos_emb
