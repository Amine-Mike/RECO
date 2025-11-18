from pathlib import Path
from typing import Optional
import warnings

from torchaudio import transforms
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio

from data import Data
from model import MLP

warnings.filterwarnings(
    "ignore", message="In 2.9, this function's implementation will be changed"
)


class Pipeline:
    def __init__(
        self,
        path: str = "data/LJSpeech-1.1",
        metadata: Optional[str] = "metadata.csv",
    ):
        self.path = Path(path)
        self.folder = self.path / Path("wavs")
        self.s_mfcc = {}
        self.metadata_path = self.path / Path(metadata)
        self.df_metadata = pd.read_csv(str(self.metadata_path), sep="|", header=None)
        self._input_size = 23

    CHARS = "abcdefghijklmnopqrstuvwxyz'?! "
    char_to_idx = {char: i + 1 for i, char in enumerate(CHARS)}
    idx_to_char = {i + 1: char for i, char in enumerate(CHARS)}

    def encode_transcription(self, msg: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx.get(c, 0) for c in msg.lower()])

    def fill_mfccs(self):
        i = 1
        max_length = 0
        for file_path in self.folder.glob("*.wav"):
            waveform, sample_rate = torchaudio.load(file_path, normalize=True)
            transform = transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=23,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )

            mfcc = transform(waveform)
            max_length = max(max_length, mfcc.shape[2])

            self.s_mfcc[file_path.stem] = Data(
                self.encode_transcription(
                    self.df_metadata[self.df_metadata[0] == file_path.stem][1].iloc[0]
                ),
                mfcc,
            )
            i += 1
            if i % 100 == 0:
                break

        for _, value in self.s_mfcc.items():
            seq = value.mfcc
            pad_size = max_length - seq.shape[2]

            value.mfcc = torch.nn.functional.pad(
                seq, (0, pad_size), mode="constant", value=0
            )

        # self._input_size = max_length

    def visualize_data(self, spectorgram: torch.Tensor):
        plt.figure(figsize=(8, 5))
        plt.imshow(
            spectorgram.squeeze().numpy(), cmap="hot", origin="lower", aspect="auto"
        )
        plt.title("MFCC")
        plt.xlabel("Frames")
        plt.ylabel("MFCC Coefficients")
        plt.colorbar()
        plt.show()

    def train_model(self):
        hidden_size = 32

        model = MLP(
            input_size=self._input_size,
            hidden_size=hidden_size,
            output_size=len(self.CHARS),
            n_layers=24,
        )
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        optim = torch.optim.Adam(model.parameters(), lr=3e-4)
        EPOCHS = 10

        for epoch in range(0, EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1} / {EPOCHS}")

            for data in self.s_mfcc.values():
                mfcc = data.mfcc
                optim.zero_grad()
                input_to_mlp = mfcc.permute(0, 2, 1).squeeze(0)

                preds = model(input_to_mlp)
                preds = preds.unsqueeze(1)
                seq_len = preds.size(0)

                input_lengths = torch.full(
                    size=(1,), fill_value=seq_len, dtype=torch.long
                )
                target_lengths = torch.tensor([data.label.size(0)], dtype=torch.long)

                loss = loss_fn(preds, data.label, input_lengths, target_lengths)
                loss.backward()

                total_loss += loss.item()
                optim.step()

            avg_loss = total_loss / len(self.s_mfcc)
            print(f"Average Loss: {avg_loss:.4f}")

    def launch_pipeline(self):
        self.fill_mfccs()

        if len(self.s_mfcc.keys()) == 0:
            raise ValueError(
                "Error while filling the mfccs, make sure that the given path contains .wav files"
            )

        # self.visualize_data(list(self.s_mfcc.values())[0].mfcc)

        self.train_model()


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.launch_pipeline()
