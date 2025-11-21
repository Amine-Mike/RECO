import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=rate, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs, training=None):
        """
        Args:
            inputs: Input tensor [batch_size, seq_len, embed_dim]
            training: Optional training flag (PyTorch handles this via model.train()/model.eval())

        Returns:
            Encoder output [batch_size, seq_len, embed_dim]
        """
        # Self-attention
        attn_output, _ = self.att(inputs, inputs, inputs, need_weights=False)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        return self.layernorm2(out1 + ffn_output)
