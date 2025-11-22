import torch
import torch.nn as nn
from embeddings import SpeechFeatureEmbedding, TokenEmbedding
from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        super().__init__()
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        encoder_layers = [
            TransformerEncoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_enc)
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)

        decoder_layers = [
            TransformerDecoder(num_hid, num_head, num_feed_forward)
            for _ in range(num_layers_dec)
        ]
        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.classifier = nn.Linear(num_hid, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to prevent gradient explosion."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)  # Smaller gain
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.constant_(self.classifier.bias, 0.0)

    def encode(self, source):
        """
        Encode the source sequence.

        Args:
            source: Source input [batch_size, source_seq_len, features]

        Returns:
            Encoded features [batch_size, source_seq_len, num_hid]
        """
        x = self.enc_input(source)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

    def decode(self, enc_out, target):
        """
        Decode with encoder output and target sequence.

        Args:
            enc_out: Encoder output [batch_size, source_seq_len, num_hid]
            target: Target sequence [batch_size, target_seq_len]

        Returns:
            Decoded features [batch_size, target_seq_len, num_hid]
        """
        y = self.dec_input(target)
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(enc_out, y)
        return y

    def forward(self, source, target):
        """
        Forward pass through the entire transformer.

        Args:
            source: Source input [batch_size, source_seq_len, features]
            target: Target sequence [batch_size, target_seq_len]

        Returns:
            Logits [batch_size, target_seq_len, num_classes]
        """
        x = self.encode(source)
        y = self.decode(x, target)
        return self.classifier(y)

    def generate(self, source, target_start_token_idx):
        """
        Performs inference over one batch of inputs using greedy decoding.

        Args:
            source: Source input [batch_size, source_seq_len, features]
            target_start_token_idx: Starting token index for generation

        Returns:
            Generated sequence [batch_size, target_maxlen]
        """
        self.eval()
        with torch.no_grad():
            bs = source.shape[0]
            enc = self.encode(source)

            dec_input = (
                torch.ones((bs, 1), dtype=torch.long, device=source.device)
                * target_start_token_idx
            )

            for _ in range(self.target_maxlen - 1):
                dec_out = self.decode(enc, dec_input)
                logits = self.classifier(dec_out)

                last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                dec_input = torch.cat([dec_input, last_token], dim=1)

            return dec_input


def compute_loss(model, source, target, criterion):
    """
    Compute loss for training/validation.

    Args:
        model: Transformer model
        source: Source input [batch_size, source_seq_len, features]
        target: Target sequence [batch_size, target_seq_len]
        criterion: Loss function (e.g., CrossEntropyLoss with ignore_index=0)

    Returns:
        loss: Computed loss value
    """
    dec_input = target[:, :-1]  # All but last token
    dec_target = target[:, 1:]  # All but first token

    preds = model(source, dec_input)  # [batch_size, seq_len, num_classes]

    preds = preds.reshape(-1, model.num_classes)  # [batch_size * seq_len, num_classes]
    dec_target = dec_target.reshape(-1)  # [batch_size * seq_len]

    loss = criterion(preds, dec_target)

    return loss


def train_step(model, batch, optimizer, criterion):
    """
    Performs one training step.

    Args:
        model: Transformer model
        batch: Dictionary with 'source' and 'target' keys
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function

    Returns:
        loss: Loss value for this batch
    """
    model.train()
    optimizer.zero_grad()

    source = batch["source"]
    target = batch["target"]

    loss = compute_loss(model, source, target, criterion)

    loss.backward()
    optimizer.step()

    return loss.item()


def test_step(model, batch, criterion):
    """
    Performs one validation/test step.

    Args:
        model: Transformer model
        batch: Dictionary with 'source' and 'target' keys
        criterion: Loss function

    Returns:
        loss: Loss value for this batch
    """
    model.eval()
    with torch.no_grad():
        source = batch["source"]
        target = batch["target"]

        loss = compute_loss(model, source, target, criterion)

    return loss.item()
