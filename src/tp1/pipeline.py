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
from CNN.model import CNN
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
        repr_type: str = "mfcc",
        repr_n_mels: int = 23,
        device: Optional[torch.device] = None,
    ):
        self.path = Path(path)
        self.folder = self.path / Path("wavs")
        self.s_rpr = {}  # Dictionary in which the keys -> Audio File name, values -> Data Type
        self.spectrograms = {}  # Dictionary in which the keys -> Audio File name, values -> Data Type
        self.metadata_path = self.path / Path(metadata)
        self.df_metadata = pd.read_csv(str(self.metadata_path), sep="|", header=None)
        self._input_size = 23
        self.repr_type = repr_type
        self.repr_n_mels = repr_n_mels
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        # Expected number of classes = len(CHARS) + 1 (blank for CTC)
        self.expected_output_size = len(self.CHARS) + 1

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

    def fill_spectrograms(self):
        """
        Create the spectrograms from a .wav file (Raw Audio), and the fill
        the dictionary for later usage
        """
        for file_path in self.folder.glob("*.wav"):
            waveform, sample_rate = torchaudio.load(file_path, normalize=True)

            mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=128
            )
            db_transform = torchaudio.transforms.AmplitudeToDB()

            mel_spectrogram = mel_spectrogram_transform(waveform)
            mel_spec_db = db_transform(mel_spectrogram)

            self.spectrograms[file_path.stem] = Data(None, mel_spec_db)

    def fill_rprs(self):
        i = 0
        max_length = 0
        for file_path in self.folder.glob("*.wav"):
            waveform, sample_rate = torchaudio.load(file_path, normalize=True)
            if self.repr_type == "mfcc":
                transform = transforms.MFCC(
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
            elif self.repr_type in {"mel", "melspectrogram"}:
                mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=400,
                    hop_length=160,
                    n_mels=self.repr_n_mels,
                )
                db_transform = torchaudio.transforms.AmplitudeToDB()
                rpr = db_transform(mel_spectrogram_transform(waveform))
            else:
                raise ValueError(f"Unknown repr_type: {self.repr_type}")
            max_length = max(max_length, rpr.shape[2])

            self.s_rpr[file_path.stem] = Data(
                self.encode_transcription(
                    self.df_metadata[self.df_metadata[0] == file_path.stem][1].iloc[0]
                ),
                rpr,
            )
            i += 1
            if i % 1000 == 0:
                break

        for _, value in self.s_rpr.items():
            seq = value.rpr
            pad_size = max_length - seq.shape[2]

            value.rpr = torch.nn.functional.pad(
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

        if self.repr_type == "mfcc":
            transform = transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self.repr_n_mels,  # Must match training
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": self.repr_n_mels,
                    "center": False,
                },
            )
            rpr = transform(waveform)

            rpr = rpr.to(self.device)
        else:
            mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=self.repr_n_mels,
            )
            db_transform = torchaudio.transforms.AmplitudeToDB()
            rpr = db_transform(mel_spectrogram_transform(waveform))

        input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)  # [Time, 23]

        with torch.no_grad():
            input_to_mlp = input_to_mlp.to(self.device)
            preds = self.model(input_to_mlp)  # [Time, Classes]
            preds = F.log_softmax(preds, dim=1)
            preds = preds.unsqueeze(1)  # [Time, 1, Classes] for consistency

        transcript = self.decode_prediction(preds.cpu())
        print(f"File: {Path(wav_path).name}")
        print(f"Prediction: {transcript}")
        return transcript

    def train_model(self):
        loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        EPOCHS = 20
        sample_answer = ""
        for epoch in range(0, EPOCHS):
            total_loss = 0
            print(f"Epoch {epoch + 1} / {EPOCHS}")
            for _, data in enumerate(self.s_rpr.values()):
                rpr = data.rpr.to(self.device)
                optim.zero_grad()
                # If model is MLP it expects [seq_len, features], not [1, channels, features, time]
                if isinstance(self.model, MLP):
                    # MFCC / mel rpr shape: [1, n_mels, time]
                    # Permute to [1, time, n_mels] then squeeze batch -> [time, n_mels]
                    input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)
                    input_to_mlp = input_to_mlp.to(self.device)
                    preds = self.model(input_to_mlp)
                    # preds shape: [time, classes], insert batch dim for CTCLoss
                    preds = preds.unsqueeze(1)
                else:
                    # CNN model expects 4D or handles 3D -> [B, C, H, W]
                    preds = self.model(rpr)
                # input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)

                preds = self.model(rpr)
                # preds = preds.unsqueeze(1)

                seq_len = preds.size(0)

                input_lengths = torch.full(
                    size=(1,), fill_value=seq_len, dtype=torch.long, device=self.device
                )
                target_lengths = torch.tensor([data.label.size(0)], dtype=torch.long)
                target_lengths = target_lengths.to(self.device)

                label = data.label.to(self.device)
                loss = loss_fn(preds, label, input_lengths, target_lengths)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                total_loss += loss.item()
                optim.step()

            # Periodic: evaluate first sample as quick sanity-check & print
            try:
                first_key = next(iter(self.s_rpr))
                print("Sanity check inference on sample:", first_key)
                sample_wav = self.s_rpr[first_key].rpr.to(self.device)
                # run the model in eval and without grad
                self.model.eval()
                with torch.no_grad():
                    preds_check = self.model(sample_wav)
                    # Move to CPU for decoding
                    transcript_check = self.decode_prediction(preds_check.cpu())
                print("  Inferred (sanity):", transcript_check)
                self.model.train()
            except StopIteration:
                pass

            avg_loss = total_loss / len(self.s_rpr)

            print(f"Average Loss: {avg_loss:.4f}")
        print(f"Sample Prediction: {sample_answer}")

    def launch_pipeline(self):
        self.fill_rprs()

        if len(self.s_rpr.keys()) == 0:
            raise ValueError(
                "Error while filling the rprs, make sure that the given path contains .wav files"
            )

        # self.visualize_data(list(self.s_rpr.values())[0].rpr)

        self.train_model()


if __name__ == "__main__":
    torch.manual_seed(42)
    INPUT_SIZE = 23
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 28  # 26 letters + space + blank
    N_LAYERS = 2
    model = MLP(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        n_layers=N_LAYERS,
    )
    # If anyone wants to use MelSpectrogram, set repr_type='mel' and repr_n_mels=128
    REPR_TYPE = "mfcc"
    REPR_N_MELS = 128

    lstm_model = CNN(
        input_chanels=1,
        output_channels=64,
        input_size=REPR_N_MELS,
        hidden_size=128,
        n_classes=OUTPUT_SIZE,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pipeline = Pipeline(
        model, repr_type=REPR_TYPE, repr_n_mels=REPR_N_MELS, device=device
    )
    pipeline.launch_pipeline()
