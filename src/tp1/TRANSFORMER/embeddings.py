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
    def __init__(self, num_dim=64, maxlen=100):
        super().__init__()
        self.conv1 = nn.Sequential()
        # self.conv2 = nn.Conv1d(
        #     in_channels=num_dim,
        # )

    def forward(self, x):
        pass
