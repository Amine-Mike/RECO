from pathlib import Path
from typing import Optional
import warnings

from torchaudio import transforms
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio

from MLP.model import MLP
from MLP.data import Data

warnings.filterwarnings(
    "ignore", message="In 2.9, this function's implementation will be changed"
)


class Pipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        path: str = "data/LJSpeech-1.1",
        metadata: Optional[str] = "metadata.csv",
    ):
        self.path = Path(path)
        self.folder = self.path / Path("wavs")
        self.s_mfcc = {}
        self.metadata_path = self.path / Path(metadata)
        self.df_metadata = pd.read_csv(str(self.metadata_path), sep="|", header=None)
        self._input_size = 23
        self.model = model

    CHARS = "abcdefghijklmnopqrstuvwxyz "
    char_to_idx = {char: i + 1 for i, char in enumerate(CHARS)}
    idx_to_char = {i + 1: char for i, char in enumerate(CHARS)}

    def encode_transcription(self, msg: str) -> torch.Tensor:
        encoded = []
        for c in msg.lower():
            if c in self.char_to_idx:
                idx = self.char_to_idx[c]
                if idx != 0:
                    encoded.append(idx)

        if len(encoded) == 0:
            return torch.tensor([1], dtype=torch.long)

        return torch.tensor(encoded, dtype=torch.long)

    def fill_mfccs(self):
        i = 0
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
            if i % 1000 == 0:
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

    def decode_prediction(self, output: torch.Tensor) -> str:
        """
        Decodes the model output (logits) into a string using Greedy Decoding.
        1. Get argmax (most likely char) for each time step.
        2. Collapse repeated characters.
        3. Drop Blank tokens (Index 0).
        """
        # output shape: [Time, 1, Classes] -> squeeze to [Time, Classes]
        output = output.squeeze(1)

        # Get the index of the highest probability character at each step
        arg_maxes = torch.argmax(output, dim=1)

        decoded_str = []
        last_idx = -1  # To track repeats

        for idx in arg_maxes:
            idx = idx.item()

            if idx != 0 and idx != last_idx:
                char = self.idx_to_char.get(idx, "")
                decoded_str.append(char)

            last_idx = idx

        return "".join(decoded_str)

    def inference(self, wav_path: str) -> str:
        self.model.eval()

        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)

        transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=23,  # Must match training
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 23,
                "center": False,
            },
        )

        mfcc = transform(waveform)

        input_to_mlp = mfcc.permute(0, 2, 1).squeeze(0)  # [Time, 23]

        with torch.no_grad():
            preds = self.model(input_to_mlp)  # [Time, Classes]
            preds = F.log_softmax(preds, dim=1)
            preds = preds.unsqueeze(1)  # [Time, 1, Classes] for consistency

        transcript = self.decode_prediction(preds)
        print(f"File: {Path(wav_path).name}")
        print(f"Prediction: {transcript}")
        return transcript

    def train_model(self):
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        EPOCHS = 10
        sample_answer = ""
        for epoch in range(0, EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1} / {EPOCHS}")
            for data in self.s_mfcc.values():
                mfcc = data.mfcc
                optim.zero_grad()
                input_to_mlp = mfcc.permute(0, 2, 1).squeeze(0)

                preds = self.model(input_to_mlp)
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
                if epoch == EPOCHS - 1:
                    sample_answer += self.decode_prediction(preds)

            avg_loss = total_loss / len(self.s_mfcc)

            print(f"Average Loss: {avg_loss:.4f}")
        print(f"Sample Prediction: {sample_answer}")

    def launch_pipeline(self):
        self.fill_mfccs()

        if len(self.s_mfcc.keys()) == 0:
            raise ValueError(
                "Error while filling the mfccs, make sure that the given path contains .wav files"
            )

        # self.visualize_data(list(self.s_mfcc.values())[0].mfcc)

        self.train_model()


if __name__ == "__main__":
    torch.manual_seed(42)
    INPUT_SIZE = 23
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 27
    N_LAYERS = 2
    model = MLP(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        n_layers=N_LAYERS,
    )
    pipeline = Pipeline(model)
    pipeline.launch_pipeline()
