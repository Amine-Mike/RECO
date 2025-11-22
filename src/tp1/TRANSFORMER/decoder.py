import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.self_att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.enc_att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.self_dropout = nn.Dropout(0.1)  # Reduced from 0.5 to prevent instability
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )

    def causal_attention_mask(self, n_dest, n_src, dtype=torch.bool):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.

        Returns 2D mask that will be broadcast across batch and heads.
        """
        i = torch.arange(n_dest).unsqueeze(1)
        j = torch.arange(n_src)
        m = i >= j - n_src + n_dest
        mask = m.to(dtype)
        return mask

    def forward(self, enc_out, target):
        """
        Args:
            enc_out: Encoder output [batch_size, src_seq_len, embed_dim]
            target: Target sequence [batch_size, tgt_seq_len, embed_dim]

        Returns:
            Decoder output [batch_size, tgt_seq_len, embed_dim]
        """
        batch_size, seq_len, _ = target.shape

        causal_mask = self.causal_attention_mask(seq_len, seq_len, dtype=torch.bool)

        # Invert mask: PyTorch masks positions where mask=True this why we put the operatio before the causal mask
        # We dont want the model to cheat so we need to use a upper matrix that will cancel
        # the possibility for the LLM to attend to the nest token and force him to attend to only seen
        # Token when computing the attention scores
        causal_mask = ~causal_mask
        causal_mask = causal_mask.to(target.device)

        target_att, _ = self.self_att(
            target, target, target, attn_mask=causal_mask, need_weights=False
        )
        target_norm = self.layernorm1(target + self.self_dropout(target_att))

        enc_out_att, _ = self.enc_att(target_norm, enc_out, enc_out, need_weights=False)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out_att) + target_norm)

        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))

        return ffn_out_norm
