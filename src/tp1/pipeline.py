from pathlib import Path
from typing import Optional

from torchaudio import transforms
import pandas as pd
import torchaudio


class Pipeline:
    def __init__(
        self,
        path: str = "data/LJSpeech-1.1",
        metadata: Optional[str] = "metadata.csv",
    ):
        self.path = Path(path)
        self.folder = self.path / Path("/wavs")
        self.s_mfcc = {}
        self.metadata_path = self.path / Path(metadata)
        self.df_metadata = pd.read_csv(str(self.metadata_path), sep="|", header=None)

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
            self.s_mfcc[str(file_path)] = mfcc

    def encode_transcription(self):
        pass

    CHARS = "abcdefghijklmnopqrstuvwxyz'?! "
    char_to_idx = {char: i + 1 for i, char in enumerate(CHARS)}
    idx_to_char = {i + 1: char for i, char in enumerate(CHARS)}

    def launch_pipeline(self):
        self.fill_mfccs()

        if self.s_mfcc.keys() is None:
            print(
                "Error while filling the mfccs, make sure that the given path contains .wav files"
            )
            return

        self.encode_transcription()


Pipeline().launch_pipeline()
