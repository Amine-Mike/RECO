import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import groupby
from MLP.model import MLP
import pandas as pd
from pathlib import Path


# Helper class to store data
class Data:
    def __init__(self, label, rpr, original_length):
        self.label = label
        self.rpr = rpr
        self.original_length = original_length


class SpeechPipeline:
    def __init__(self, folder, df_metadata, model, idx_to_char):
        self.folder = Path(folder)
        self.df_metadata = df_metadata
        self.model = model
        self.idx_to_char = idx_to_char
        self.s_rpr = {}

        # Define transform ONCE here to ensure consistency
        self.transform = transforms.MFCC(
            sample_rate=16000,  # Ensure this matches your wav files
            n_mfcc=23,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 23,
                "center": False,
            },
        )

    def fill_rprs(self):
        i = 0
        max_length = 0

        # 1. Load and Store (First Pass)
        for file_path in self.folder.glob("*.wav"):
            waveform, sample_rate = torchaudio.load(file_path, normalize=True)

            # Check Sample Rate consistency
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

            rpr = self.transform(waveform)

            # Track max length for padding later
            max_length = max(max_length, rpr.shape[2])

            # Save the ORIGINAL length (Time dimension)
            original_len = rpr.shape[2]

            transcript_tensor = self.encode_transcription(
                self.df_metadata[self.df_metadata[0] == file_path.stem][1].iloc[0]
            )

            self.s_rpr[file_path.stem] = Data(
                label=transcript_tensor, rpr=rpr, original_length=original_len
            )

            i += 1
            if i % 1000 == 0:
                break

        # 2. Pad (Second Pass)
        for _, value in self.s_rpr.items():
            seq = value.rpr
            pad_size = max_length - seq.shape[2]
            # Pad the time dimension (last dimension)
            value.rpr = torch.nn.functional.pad(
                seq, (0, pad_size), mode="constant", value=0
            )

    def decode_prediction(self, output: torch.Tensor) -> str:
        """
        Improved Pythonic Greedy Decoder
        """
        # output: [Time, Classes] (After squeeze)
        output = output.squeeze(1)
        arg_maxes = torch.argmax(output, dim=1).tolist()

        decoded_str = []

        # groupby collapses repeated characters (A, A, B -> A, B)
        for char_idx, _ in groupby(arg_maxes):
            if char_idx != 0:  # Remove Blanks (Index 0)
                decoded_str.append(self.idx_to_char.get(char_idx, ""))

        return "".join(decoded_str)

    def train_model(self):
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        EPOCHS = 10

        self.model.train()  # Ensure training mode
        for epoch in range(EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1} / {EPOCHS}")

            for data in self.s_rpr.values():
                rpr = data.rpr
                optim.zero_grad()

                # [Batch, N_MFCC, Time] -> [Time, Batch, N_MFCC]
                # Assuming Batch size is 1 for this loop
                input_to_mlp = rpr.permute(2, 0, 1).squeeze(1)

                preds = self.model(input_to_mlp)  # [Time, Classes]

                # CTC expects Log Softmax
                preds = F.log_softmax(preds, dim=1)

                # CTC Loss inputs: [Time, Batch, Classes]
                preds = preds.unsqueeze(1)

                # --- THE FIX IS HERE ---
                # Use the ORIGINAL length, not the padded length
                input_lengths = torch.tensor([data.original_length], dtype=torch.long)
                target_lengths = torch.tensor([data.label.size(0)], dtype=torch.long)

                loss = loss_fn(preds, data.label, input_lengths, target_lengths)
                loss.backward()
                total_loss += loss.item()
                optim.step()

                print(self.decode_prediction(preds))

            print(f"Average Loss: {total_loss / len(self.s_rpr):.4f}")

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

        rpr = transform(waveform)

        input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)  # [Time, 23]

        with torch.no_grad():
            preds = self.model(input_to_mlp)  # [Time, Classes]
            preds = F.log_softmax(preds, dim=1)
            preds = preds.unsqueeze(1)  # [Time, 1, Classes] for consistency

        transcript = self.decode_prediction(preds)
        print(f"File: {Path(wav_path).name}")
        print(f"Prediction: {transcript}")
        return transcript

    def launch_pipeline(self):
        self.fill_rprs()

        if len(self.s_rpr.keys()) == 0:
            raise ValueError(
                "Error while filling the rprs, make sure that the given path contains .wav files"
            )

        # self.visualize_data(list(self.s_rpr.values())[0].rpr)

        self.train_model()

    # ... encode_transcription and inference would remain similar ...


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
    path = Path("data/LJSpeech-1.1")
    metadata = "metadata.csv"
    CHARS = "abcdefghijklmnopqrstuvwxyz "
    char_to_idx = {char: i + 1 for i, char in enumerate(CHARS)}
    idx_to_char = {i + 1: char for i, char in enumerate(CHARS)}
    metadata_path = path / Path(metadata)
    folder = Path("data/LJSpeech-1.1") / Path("wavs")
    df_metadata = pd.read_csv(str(metadata_path), sep="|", header=None)
    pipeline = SpeechPipeline(
        folder=folder, df_metadata=df_metadata, model=model, idx_to_char=idx_to_char
    )
    pipeline.launch_pipeline()
