from typing import Optional

import torch
import torchaudio

from CNN.model import CNN
from pipeline_common import (
    fill_rprs_from_folder,
    CHARS,
    decode_prediction,
)


class CNNPipeline:
    def __init__(
        self,
        model: CNN,
        folder: str = "data/LJSpeech-1.1/wavs",
        metadata: str = "data/LJSpeech-1.1/metadata.csv",
        repr_type: str = "mel",
        repr_n_mels: int = 128,
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
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.s_rpr = {}
        self.val_sample = None
        self.val_sample_name = None
        self.expected_output_size = len(CHARS) + 1

    def fill_rprs(self):
        print("Loading validation sample...")
        val_samples = fill_rprs_from_folder(
            self.folder,
            self.metadata,
            repr_type=self.repr_type,
            repr_n_mels=self.repr_n_mels,
            max_samples=1,
            pad_sequences=False,
        )
        if val_samples:
            self.val_sample_name = next(iter(val_samples))
            self.val_sample = val_samples[self.val_sample_name]
            print(f"Validation sample: {self.val_sample_name}")

        print(f"Loading training samples (max_samples={self.max_samples})...")
        self.s_rpr = fill_rprs_from_folder(
            self.folder,
            self.metadata,
            repr_type=self.repr_type,
            repr_n_mels=self.repr_n_mels,
            max_samples=self.max_samples,
            pad_sequences=False,
            skip_samples=1,
        )
        print(f"Loaded {len(self.s_rpr)} training samples")

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

        # rpr shape: [1, n_mels, time] -> add channel dim -> [1, 1, n_mels, time]
        rpr = rpr.unsqueeze(0) if rpr.dim() == 3 else rpr
        rpr = rpr.unsqueeze(1) if rpr.dim() == 3 else rpr
        rpr = rpr.to(self.device)

        with torch.no_grad():
            preds = self.model(rpr)

        return decode_prediction(preds.cpu())

    def train_model(self, epochs: int = 20, lr: float = 3e-4):
        print(f"Starting training on {self.device} Device")
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            self.model.train()
            total_loss = 0.0
            for _, data in enumerate(self.s_rpr.values()):
                rpr = data.rpr.to(self.device)
                label = data.label.to(self.device)

                if rpr.dim() == 3:
                    rpr = rpr.unsqueeze(0)
                if rpr.dim() == 3:
                    rpr = rpr.unsqueeze(1)

                preds = self.model(rpr)  # [time, batch, classes]

                input_lengths = torch.tensor(
                    [preds.size(0)], dtype=torch.long, device=self.device
                )
                target_lengths = torch.tensor(
                    [label.size(0)], dtype=torch.long, device=self.device
                )

                opt.zero_grad()
                loss = loss_fn(preds, label, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                opt.step()

                total_loss += loss.item()

            # Validation inference on held-out sample
            if self.val_sample is not None:
                print(f"Validation inference on sample: {self.val_sample_name}")
                val_rpr = self.val_sample.rpr.to(self.device)

                # ensure shape: [batch, ch, n_mels, time]
                if val_rpr.dim() == 3:
                    val_rpr = val_rpr.unsqueeze(0)
                if val_rpr.dim() == 3:
                    val_rpr = val_rpr.unsqueeze(1)

                self.model.eval()
                with torch.no_grad():
                    preds_check = self.model(val_rpr)
                    transcript_check = decode_prediction(preds_check.cpu())
                print("  Predicted:", transcript_check)
                self.model.train()

            avg = total_loss / len(self.s_rpr) if len(self.s_rpr) else 0
            print(f"Avg loss: {avg:.4f}")

    def launch_pipeline(self):
        self.fill_rprs()
        if len(self.s_rpr) == 0:
            raise RuntimeError("No wav files were found in the provided folder")
        self.train_model(epochs=50)


if __name__ == "__main__":
    INPUT_SIZE = 128
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = len(CHARS) + 1
    MODEL_TYPE = "GRU"
    model = CNN(
        input_chanels=1,
        output_channels=64,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        n_classes=OUTPUT_SIZE,
        model_type=MODEL_TYPE,
    )
    pipeline = CNNPipeline(
        model,
        repr_type="mel",
        repr_n_mels=INPUT_SIZE,
        folder="data/LJSpeech-1.1/wavs",
    )
    print(f"Launching CNN pipeline with model type: {MODEL_TYPE}")
    pipeline.launch_pipeline()
