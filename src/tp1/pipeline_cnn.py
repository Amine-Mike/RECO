from pathlib import Path as PathType
from typing import Optional

import pandas as pd
import torch
import torchaudio
from CNN.model import CNN
from pipeline_common import (
    CHARS,
    decode_prediction,
    fill_rprs_from_folder,
    idx_to_char,
)
from torch.utils.data import DataLoader, Dataset


class CNNDataset(Dataset):
    """Dataset for CNN training with variable length sequences."""

    def __init__(self, data_dict):
        """
        Args:
            data_dict: Dictionary from fill_rprs_from_folder {name: Data(label, rpr)}
        """
        self.data = list(data_dict.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            "rpr": data.rpr,
            "label": data.label,
        }


def collate_fn_cnn(batch):
    """
    Collate function for CNN that pads to max length in batch.

    Args:
        batch: List of dicts with 'rpr' and 'label'

    Returns:
        Dictionary with batched and padded tensors
    """
    max_time = max(item["rpr"].shape[2] for item in batch)

    # Pad all sequences to max_time
    rprs = []
    labels = []
    input_lengths = []
    target_lengths = []

    for item in batch:
        rpr = item["rpr"]
        label = item["label"]

        # Pad rpr: [1, n_mels, time] -> [1, n_mels, max_time]
        pad_size = max_time - rpr.shape[2]
        if pad_size > 0:
            rpr = torch.nn.functional.pad(rpr, (0, pad_size), mode="constant", value=0)

        rprs.append(rpr)
        labels.append(label)
        input_lengths.append(rpr.shape[2])  # Will be computed after model forward
        target_lengths.append(label.shape[0])

    # Stack into batch
    rprs = torch.cat(rprs, dim=0)  # [batch, n_mels, max_time]

    # Pad labels to same length
    max_label_len = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        pad_size = max_label_len - len(label)
        if pad_size > 0:
            label = torch.nn.functional.pad(
                label, (0, pad_size), mode="constant", value=0
            )
        padded_labels.append(label)

    labels = torch.stack(padded_labels)  # [batch, max_label_len]
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return {
        "rpr": rprs,
        "labels": labels,
        "target_lengths": target_lengths,
    }


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
        self.train_data = {}
        self.val_data = {}
        self.expected_output_size = len(CHARS) + 1

    def fill_rprs(self, train_split: float = 0.99):
        """Load data and split into train/val sets."""
        print(f"Loading samples (max_samples={self.max_samples})...")
        all_data = fill_rprs_from_folder(
            self.folder,
            self.metadata,
            repr_type=self.repr_type,
            repr_n_mels=self.repr_n_mels,
            max_samples=self.max_samples,
            pad_sequences=False,
            skip_samples=0,
        )

        all_items = list(all_data.items())
        split_idx = int(len(all_items) * train_split)

        self.train_data = dict(all_items[:split_idx])
        self.val_data = dict(all_items[split_idx:])

        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")

    def _display_predictions(self, val_batch, num_samples: int = 4):
        """Display model predictions on validation batch."""
        self.model.eval()
        with torch.no_grad():
            rpr = val_batch["rpr"].to(self.device)
            labels = val_batch["labels"].cpu()
            target_lengths = val_batch["target_lengths"].cpu()

            if rpr.dim() == 3:
                rpr = rpr.unsqueeze(1)

            preds = self.model(rpr)

            print("\n" + "=" * 80)
            for i in range(min(num_samples, len(labels))):
                target_indices = labels[i, : target_lengths[i]].tolist()
                target_chars = [
                    idx_to_char.get(idx, "") for idx in target_indices if idx != 0
                ]
                target_text = "".join(target_chars)

                pred_text = decode_prediction(preds[:, i : i + 1, :].cpu())

                print(f"Target:     {target_text}")
                print(f"Prediction: {pred_text}")
                print("-" * 80)
            print("=" * 80 + "\n")

    def _get_text_from_metadata(self, sample_name: str) -> str:
        """Get ground truth text for a sample from metadata."""
        try:
            metadata_path = PathType(self.metadata)
            if metadata_path.exists():
                df = pd.read_csv(str(metadata_path), sep="|", header=None)
                row = df[df[0] == sample_name]
                if not row.empty:
                    return row[1].iloc[0].lower()
        except (FileNotFoundError, KeyError, IndexError, ValueError):
            pass
        return "[ground truth not available]"

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

    def train_model(
        self,
        epochs: int = 20,
        lr: float = 3e-4,
        batch_size: int = 32,
        num_workers: int = 4,
        display_every: int = 5,
        checkpoint_dir: str = "checkpoints_cnn",
    ):
        print(f"Starting training on {self.device} Device")
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        checkpoint_dir = PathType(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        train_dataset = CNNDataset(self.train_data)
        val_dataset = CNNDataset(self.val_data)

        use_pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_cnn,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn_cnn,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

        print(f"Training with {len(train_dataset)} samples, batch_size={batch_size}")

        val_batch = next(iter(val_loader))

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                rpr = batch["rpr"].to(self.device)
                labels = batch["labels"].to(self.device)
                target_lengths = batch["target_lengths"].to(self.device)

                # Ensure shape: [batch, ch, n_mels, time]
                if rpr.dim() == 3:
                    rpr = rpr.unsqueeze(1)

                preds = self.model(rpr)  # [time, batch, classes]

                # Compute input lengths for each sample in batch
                input_lengths = torch.full(
                    (preds.size(1),),
                    preds.size(0),
                    dtype=torch.long,
                    device=self.device,
                )

                # Flatten labels for CTC loss
                labels_flat = []
                for i in range(labels.size(0)):
                    labels_flat.append(labels[i, : target_lengths[i]])
                labels_concat = torch.cat(labels_flat)

                opt.zero_grad()
                loss = loss_fn(preds, labels_concat, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"\nWarning: NaN/Inf loss at epoch {epoch + 1}, batch {batch_idx}"
                    )
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=2.0
                )
                opt.step()

                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] "
                        f"Batch [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"GradNorm: {grad_norm:.3f}",
                        end="\r",
                    )

            avg_train_loss = train_loss / len(train_loader) if len(train_loader) else 0

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    rpr = batch["rpr"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    target_lengths = batch["target_lengths"].to(self.device)

                    if rpr.dim() == 3:
                        rpr = rpr.unsqueeze(1)

                    preds = self.model(rpr)

                    input_lengths = torch.full(
                        (preds.size(1),),
                        preds.size(0),
                        dtype=torch.long,
                        device=self.device,
                    )

                    labels_flat = []
                    for i in range(labels.size(0)):
                        labels_flat.append(labels[i, : target_lengths[i]])
                    labels_concat = torch.cat(labels_flat)

                    loss = loss_fn(preds, labels_concat, input_lengths, target_lengths)
                    val_loss += loss.item()

            avg_val_loss = val_loss / (len(val_loader) + 1e-8)

            print(
                f"\nEpoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f}"
            )

            if (epoch + 1) % display_every == 0:
                self._display_predictions(val_batch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    checkpoint_path,
                )
                print(f"Saved best model to {checkpoint_path}")

            if (epoch + 1) % 10 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                # Save additional meta info so we can detect model type later
                lstm = getattr(self.model, "lstm", None)
                model_type = (
                    lstm.__class__.__name__
                    if lstm is not None
                    else self.model.__class__.__name__
                )
                bidir = (
                    getattr(lstm, "bidirectional", False) if lstm is not None else False
                )
                hidden = (
                    getattr(lstm, "hidden_size", None) if lstm is not None else None
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "model_type": ("BI-" + model_type if bidir else model_type),
                        "hidden_size": hidden,
                        "repr_n_mels": self.repr_n_mels,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

    def launch_pipeline(self, epochs: int = 50, batch_size: int = 32):
        self.fill_rprs()
        if len(self.train_data) == 0:
            raise RuntimeError("No wav files were found in the provided folder")
        self.train_model(epochs=epochs, batch_size=batch_size)


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
