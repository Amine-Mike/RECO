from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchaudio
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class VectorizeChar:
    """Character vectorization for text targets."""

    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

    def __call__(self, text):
        """
        Vectorize text to indices with start/end tokens and padding.

        Args:
            text: Input text string

        Returns:
            List of indices with padding
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        """Return the vocabulary list."""
        return self.vocab

    def decode(self, indices):
        """
        Decode indices back to text.

        Args:
            indices: Tensor or list of indices

        Returns:
            Decoded string
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        text = ""
        for idx in indices:
            if idx == 0:
                continue
            if idx == 3:
                break
            text += self.idx_to_char.get(idx, "#")

        text = text.replace("<", "").replace("-", "")
        return text


def path_to_audio(path, pad_len=2754):
    """
    Load and preprocess audio file to STFT spectrogram.

    Args:
        path: Path to audio file
        pad_len: Length to pad/truncate to (default: 2754 for 10 seconds)

    Returns:
        Normalized spectrogram [time, freq]
    """
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    audio = waveform.squeeze(0)  # Remove channel dimension

    stfts = torch.stft(
        audio,
        n_fft=256,
        hop_length=80,
        win_length=200,
        window=torch.hann_window(200),
        return_complex=True,
    )

    x = torch.abs(stfts).pow(0.5)
    x = x.transpose(0, 1)  # [time, freq]

    means = x.mean(dim=0, keepdim=True)
    stddevs = x.std(dim=0, keepdim=True)
    x = (x - means) / (stddevs + 1e-8)

    audio_len = x.shape[0]
    if audio_len < pad_len:
        padding = torch.zeros((pad_len - audio_len, x.shape[1]))
        x = torch.cat([x, padding], dim=0)
    else:
        x = x[:pad_len, :]

    return x


class TransformerDataset(Dataset):
    """Dataset for Transformer speech recognition."""

    def __init__(
        self,
        audio_paths: List[str],
        texts: List[str],
        vectorizer: VectorizeChar,
        pad_len: int = 2754,
    ):
        """
        Args:
            audio_paths: List of audio file paths
            texts: List of corresponding transcriptions
            vectorizer: VectorizeChar instance for text encoding
            pad_len: Audio padding length
        """
        self.audio_paths = audio_paths
        self.texts = texts
        self.vectorizer = vectorizer
        self.pad_len = pad_len

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = path_to_audio(self.audio_paths[idx], self.pad_len)

        text_indices = self.vectorizer(self.texts[idx])
        text = torch.tensor(text_indices, dtype=torch.long)

        return {"source": audio, "target": text}


def get_data_from_folder(
    wav_folder: str, metadata_path: str, max_samples: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Load audio paths and texts from folder and metadata.

    Args:
        wav_folder: Path to folder containing .wav files
        metadata_path: Path to metadata CSV file
        max_samples: Optional limit on number of samples

    Returns:
        List of dicts with 'audio' and 'text' keys
    """

    wav_folder = Path(wav_folder)
    metadata_path = Path(metadata_path)

    df = pd.read_csv(
        metadata_path, sep="|", header=None, names=["id", "text", "normalized"]
    )

    data = []
    skipped = 0
    for i, row in df.iterrows():
        audio_path = wav_folder / f"{row['id']}.wav"

        if audio_path.exists():
            text = row["normalized"] if pd.notna(row["normalized"]) else row["text"]

            if pd.isna(text) or (isinstance(text, float) and not isinstance(text, str)):
                skipped += 1
                continue

            data.append(
                {
                    "audio": str(audio_path),
                    "text": str(text),  # Ensure it's a string
                }
            )

        if max_samples and len(data) >= max_samples:
            break

    if skipped > 0:
        print(f"  Warning: Skipped {skipped} samples with invalid text")

    return data


def create_dataloaders(
    data: List[Dict[str, str]],
    vectorizer: VectorizeChar,
    batch_size: int = 64,
    train_split: float = 0.99,
    pad_len: int = 2754,
    num_workers: int = 4,
):
    """
    Create train and validation dataloaders.

    Args:
        data: List of dicts with 'audio' and 'text' keys
        vectorizer: VectorizeChar instance
        batch_size: Batch size
        train_split: Fraction of data for training
        pad_len: Audio padding length
        num_workers: Number of workers for dataloader

    Returns:
        train_loader, val_loader
    """
    # Split data
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Extract paths and texts
    train_paths = [d["audio"] for d in train_data]
    train_texts = [d["text"] for d in train_data]
    val_paths = [d["audio"] for d in val_data]
    val_texts = [d["text"] for d in val_data]

    # Create datasets
    train_dataset = TransformerDataset(train_paths, train_texts, vectorizer, pad_len)
    val_dataset = TransformerDataset(val_paths, val_texts, vectorizer, pad_len)

    # Check if pin_memory should be used (not supported on MPS)
    use_pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader


class CustomSchedule:
    """Learning rate schedule with linear warmup and decay."""

    def __init__(
        self,
        optimizer,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0

    def calculate_lr(self, epoch):
        """Linear warmup - linear decay."""
        if epoch < self.warmup_epochs:
            # Warmup phase
            warmup_lr = (
                self.init_lr
                + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1))
                * epoch
            )
            return warmup_lr
        else:
            # Decay phase
            decay_lr = max(
                self.final_lr,
                self.lr_after_warmup
                - (epoch - self.warmup_epochs)
                * (self.lr_after_warmup - self.final_lr)
                / self.decay_epochs,
            )
            return decay_lr

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        epoch = self.current_step / self.steps_per_epoch
        lr = self.calculate_lr(epoch)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


def display_predictions(
    model,
    batch,
    vectorizer: VectorizeChar,
    target_start_token_idx=2,
    target_end_token_idx=3,
    device="cuda",
    num_samples=4,
):
    """
    Display model predictions for a batch.

    Args:
        model: Transformer model
        batch: Batch dict with 'source' and 'target'
        vectorizer: VectorizeChar instance
        target_start_token_idx: Start token index
        target_end_token_idx: End token index
        device: Device to run on
        num_samples: Number of samples to display
    """
    model.eval()
    with torch.no_grad():
        source = batch["source"].to(device)
        target = batch["target"].cpu().numpy()

        # Generate predictions
        preds = model.generate(source, target_start_token_idx)
        preds = preds.cpu().numpy()

        print("\n" + "=" * 80)
        for i in range(min(num_samples, len(target))):
            # Decode target
            target_text = vectorizer.decode(target[i])

            # Decode prediction
            prediction = ""
            for idx in preds[i]:
                if idx == target_end_token_idx:
                    break
                prediction += vectorizer.idx_to_char.get(idx, "#")
            prediction = prediction.replace("<", "").replace("-", "")

            print(f"Target:     {target_text}")
            print(f"Prediction: {prediction}")
            print("-" * 80)
        print("=" * 80 + "\n")


def train_transformer(
    model,
    train_loader,
    val_loader,
    vectorizer: VectorizeChar,
    epochs=100,
    device="cuda",
    checkpoint_dir="checkpoints",
    display_every=5,
):
    """
    Train the transformer model.

    Args:
        model: Transformer model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        vectorizer: VectorizeChar instance
        epochs: Number of epochs
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        display_every: Display predictions every N epochs
    """
    model = model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    optimizer = Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CustomSchedule(
        optimizer,
        init_lr=0.00001,
        lr_after_warmup=0.0003,
        final_lr=0.00001,
        warmup_epochs=20,
        decay_epochs=80,
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)

    val_batch = next(iter(val_loader))

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            dec_input = target[:, :-1]
            dec_target = target[:, 1:]

            optimizer.zero_grad()
            preds = model(source, dec_input)

            loss = criterion(
                preds.reshape(-1, model.num_classes), dec_target.reshape(-1)
            )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            if torch.isnan(loss) or torch.isinf(loss) or grad_norm > 100.0:
                print(
                    f"\nWarning: NaN/Inf loss or extreme gradients detected at epoch {epoch + 1}, batch {batch_idx}"
                )
                print(f"Loss: {loss.item()}, Grad Norm: {grad_norm}")
                print("Stopping training to prevent further issues.")
                print(f"Last valid checkpoint: {checkpoint_dir / 'best_model.pt'}")
                return
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"GradNorm: {grad_norm:.3f} "
                    f"LR: {current_lr:.6f}",
                    end="\r",
                )

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                source = batch["source"].to(device)
                target = batch["target"].to(device)

                dec_input = target[:, :-1]
                dec_target = target[:, 1:]

                preds = model(source, dec_input)
                loss = criterion(
                    preds.reshape(-1, model.num_classes), dec_target.reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / (len(val_loader) + 1)

        print(
            f"\nEpoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if (epoch + 1) % display_every == 0:
            display_predictions(model, val_batch, vectorizer, device=device)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"✓ Saved best model to {checkpoint_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")
