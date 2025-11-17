from pathlib import Path
from typing import Optional
import warnings

import torch
from torchaudio import transforms
import pandas as pd
import torchaudio

from data import Data

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

    CHARS = "abcdefghijklmnopqrstuvwxyz'?! "
    char_to_idx = {char: i + 1 for i, char in enumerate(CHARS)}
    idx_to_char = {i + 1: char for i, char in enumerate(CHARS)}

    def encode_transcription(self, msg: str):
        return torch.tensor([self.char_to_idx.get(c, 0) for c in msg.lower()])

    def fill_mfccs(self):
        for file_path in self.folder.glob("*.wav"):
            waveform, sample_rate = torchaudio.load(file_path, normalize=True)
            transform = transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 23,
                    "center": False,
                },
            )
            mfcc = transform(waveform)
            self.s_mfcc[file_path.stem] = Data(
                self.encode_transcription(
                    self.df_metadata[self.df_metadata[0] == file_path.stem][1].iloc[0]
                ),
                mfcc,
            )

    def launch_pipeline(self):
        self.fill_mfccs()

        if len(self.s_mfcc.keys()) == 0:
            raise ValueError(
                "Error while filling the mfccs, make sure that the given path contains .wav files"
            )
