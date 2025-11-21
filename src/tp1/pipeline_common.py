from pathlib import Path
from typing import Dict, Callable, Optional
import warnings
import pandas as pd

import torch
import torchaudio
from torchaudio import transforms

from MLP.data import Data

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


CHARS = "abcdefghijklmnopqrstuvwxyz "
char_to_idx = {char: i + 1 for i, char in enumerate(CHARS)}
idx_to_char = {i + 1: char for i, char in enumerate(CHARS)}


def encode_transcription(msg: str) -> torch.Tensor:
    encoded = []
    for c in msg.lower():
        if c in char_to_idx:
            idx = char_to_idx[c]
            if idx != 0:
                encoded.append(idx)

    if len(encoded) == 0:
        return torch.tensor([1], dtype=torch.long)

    return torch.tensor(encoded, dtype=torch.long)


def decode_prediction(output: torch.Tensor) -> str:
    # output shape: [Time, 1, Classes] -> squeeze to [Time, Classes]
    output = output.squeeze(1)
    arg_maxes = torch.argmax(output, dim=1)

    decoded_str = []
    last_idx = -1

    for idx in arg_maxes:
        idx = idx.item()
        if idx != 0 and idx != last_idx:
            char = idx_to_char.get(idx, "")
            decoded_str.append(char)
        last_idx = idx

    return "".join(decoded_str)


def _build_mfcc_transform(sample_rate: int, n_mfcc: int) -> Callable:
    return transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": n_mfcc,
            "center": False,
        },
    )


def _build_mel_transform(sample_rate: int, n_mels: int) -> Callable:
    mel_spec = transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels
    )
    db = transforms.AmplitudeToDB()

    def _apply(waveform: torch.Tensor) -> torch.Tensor:
        return db(mel_spec(waveform))

    return _apply


def fill_rprs_from_folder(
    folder: str,
    metadata_path: str,
    repr_type: str = "mfcc",
    repr_n_mels: int = 23,
    max_samples: Optional[int] = None,
    pad_sequences: bool = True,
    skip_samples: int = 0,
) -> Dict[str, Data]:
    """
    Scans the folder for wavs and builds representations. Returns dict mapping filename stem -> Data(label, rpr)
    If pad_sequences=True (for MLP), pads all sequences to the longest length.
    If pad_sequences=False (for CNN+LSTM), keeps variable lengths.

    Args:
        skip_samples: Number of samples to skip at the beginning (useful for train/val split)
    """
    folder = Path(folder)
    metadata_path = Path(metadata_path)
    df_metadata = None
    if metadata_path.exists():
        df_metadata = pd.read_csv(str(metadata_path), sep="|", header=None)

    s_rpr = {}
    max_length = 0 if pad_sequences else None
    i = 0

    for file_path in folder.glob("*.wav"):
        if i < skip_samples:
            i += 1
            continue

        if (i - skip_samples) % 10 == 0:
            print(f"  Loading sample {i - skip_samples}...", end="\r")

        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        if i % 10 == 0:
            print(f"  Loading sample {i}...", end="\r")
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)

        if repr_type == "mfcc":
            transform = _build_mfcc_transform(sample_rate, repr_n_mels)
            rpr = transform(waveform)
        elif repr_type in {"mel", "melspectrogram"}:
            transform = _build_mel_transform(sample_rate, repr_n_mels)
            rpr = transform(waveform)
        else:
            raise ValueError(f"Unknown repr_type: {repr_type}")

        if pad_sequences:
            max_length = max(max_length, rpr.shape[2])

        label_tensor = None
        if df_metadata is not None:
            try:
                label = df_metadata[df_metadata[0] == file_path.stem][1].iloc[0]
                label_tensor = encode_transcription(label)
            except (KeyError, IndexError):
                label_tensor = torch.tensor([1], dtype=torch.long)

        s_rpr[file_path.stem] = Data(label_tensor, rpr)

        i += 1
        if max_samples is not None and (i - skip_samples) >= max_samples:
            break

    if pad_sequences:
        print(
            f"\n  Padding sequences to max_length={max_length} (for MLP fixed-size input)..."
        )
        for _, value in s_rpr.items():
            seq = value.rpr
            pad_size = max_length - seq.shape[2]
            if pad_size > 0:
                value.rpr = torch.nn.functional.pad(
                    seq, (0, pad_size), mode="constant", value=0
                )
    else:
        print(f"\n  Loaded {len(s_rpr)} samples (variable lengths for CNN+LSTM)")

    return s_rpr
