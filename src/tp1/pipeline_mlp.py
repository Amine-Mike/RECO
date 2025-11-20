from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio

from MLP.model import MLP
from MLP.data import Data
from pipeline_common import (
    fill_rprs_from_folder,
    CHARS,
    encode_transcription,
    decode_prediction,
)


class MLPPipeline:
    def __init__(
        self,
        model: MLP,
        folder: str = "data/LJSpeech-1.1/wavs",
        metadata: str = "data/LJSpeech-1.1/metadata.csv",
        repr_type: str = "mfcc",
        repr_n_mels: int = 23,
        device: Optional[torch.device] = None,
        max_samples: Optional[int] = None,
    ):
        self.model = model
        self.folder = folder
        self.metadata = metadata
        self.repr_type = repr_type
        self.repr_n_mels = repr_n_mels
        self.max_samples = max_samples
        self.device = (
            device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.s_rpr = {}

        # CTCLoss expects classes + 1 blank token
        self.expected_output_size = len(CHARS) + 1

    def fill_rprs(self) -> None:
        print(f"Loading audio files (max_samples={self.max_samples})...")
        self.s_rpr = fill_rprs_from_folder(
            self.folder, self.metadata, repr_type=self.repr_type, repr_n_mels=self.repr_n_mels,
            max_samples=self.max_samples, pad_sequences=True
        )
        print(f"Loaded {len(self.s_rpr)} samples")

    def inference(self, wav_path: str) -> str:
        self.model.eval()
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)

        if self.repr_type == "mfcc":
            transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self.repr_n_mels,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": self.repr_n_mels,
                    "center": False,
                },
            )
            rpr = transform(waveform)
        else:
            mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=self.repr_n_mels,
            )
            db_transform = torchaudio.transforms.AmplitudeToDB()
            rpr = db_transform(mel_spectrogram_transform(waveform))

        # shape: [1, n_mels, time] -> permute to [time, n_mels]
        input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)
        input_to_mlp = input_to_mlp.to(self.device)

        with torch.no_grad():
            preds = self.model(input_to_mlp)
            if preds.dim() == 2:
                preds = preds.unsqueeze(1)

        return decode_prediction(preds.cpu())

    def train_model(self, epochs: int = 20, lr: float = 3e-4):
        print(f"Starting training on {self.device} Device")
        # set device
        self.model.to(self.device)

        # Ensure model's input size matches repr
        first_linear = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            raise RuntimeError("Cannot find Linear layer in the MLP model")

        if first_linear.in_features != self.repr_n_mels:
            raise ValueError(
                f"MLP's first Linear expects {first_linear.in_features} features but repr_n_mels={self.repr_n_mels}"
            )

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            self.model.train()
            total_loss = 0.0
            for i, data in enumerate(self.s_rpr.values()):
                rpr = data.rpr.to(self.device)
                label = data.label.to(self.device)

                # permute to [time, features]
                input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)
                preds = self.model(input_to_mlp)  # [time, classes]
                preds = preds.unsqueeze(1)  # [time, batch, classes]

                input_lengths = torch.tensor([preds.size(0)], dtype=torch.long, device=self.device)
                target_lengths = torch.tensor([label.size(0)], dtype=torch.long, device=self.device)

                opt.zero_grad()
                loss = loss_fn(preds, label, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                opt.step()

                total_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(self.s_rpr)} samples, loss={loss.item():.4f}")

            try:
                first_key = next(iter(self.s_rpr))
                print("Sanity check inference on sample:", first_key)
                sample_rpr = self.s_rpr[first_key].rpr.to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    # permute to [time, features] for MLP
                    input_to_mlp = sample_rpr.permute(0, 2, 1).squeeze(0)
                    preds_check = self.model(input_to_mlp)
                    if preds_check.dim() == 2:
                        preds_check = preds_check.unsqueeze(1)
                    transcript_check = decode_prediction(preds_check.cpu())
                print("  Inferred (sanity):", transcript_check)
                self.model.train()
            except StopIteration:
                pass

            avg = total_loss / len(self.s_rpr) if len(self.s_rpr) else 0
            print(f"Avg loss: {avg:.4f}")

    def launch_pipeline(self):
        self.fill_rprs()
        if len(self.s_rpr) == 0:
            raise RuntimeError("No wav files were found in the provided folder")
        self.train_model()


if __name__ == "__main__":
    from pathlib import Path
    import torch

    INPUT_SIZE = 23
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = len(CHARS) + 1
    N_LAYERS = 2

    mlp_model = MLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, n_layers=N_LAYERS)
    pipeline = MLPPipeline(mlp_model, repr_type="mfcc", repr_n_mels=INPUT_SIZE, folder="data/LJSpeech-1.1/wavs")
    print("Launching MLP pipeline")
    pipeline.launch_pipeline()
