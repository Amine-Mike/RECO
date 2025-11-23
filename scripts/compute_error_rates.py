import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from src.tp1.pipeline_common import (
    CHARS,
    decode_prediction,
    fill_rprs_from_folder,
    idx_to_char,
)
from src.tp1.CNN.model import CNN
from src.tp1.pipeline_cnn import CNNDataset, collate_fn_cnn
from src.tp1.pipeline_transformer import VectorizeChar, get_data_from_folder
from src.tp1.TRANSFORMER.model import Transformer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src" / "tp1"))


def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def compute_cer(pred: str, ref: str) -> Tuple[int, int]:
    pred_chars = list(pred)
    ref_chars = list(ref)
    d = levenshtein(pred_chars, ref_chars)
    return d, len(ref_chars)


def compute_wer(pred: str, ref: str) -> Tuple[int, int]:
    pred_words = pred.strip().split()
    ref_words = ref.strip().split()
    d = levenshtein(pred_words, ref_words)
    return d, len(ref_words)


def cnn_evaluate(
    checkpoint_path: str,
    data_dir: str,
    max_samples: int = None,
    device: str = "cuda",
    batch_size: int = 32,
    repr_n_mels: int = 128,
    hidden_size: int = 128,
    model_type: str = "BI-LSTM",
    output_channels: int = 64,
    train_split: float = 0.99,
    num_workers: int = 4,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    # If checkpoint contains model_type / hidden_size, recreate the model accordingly
    ckpt_model_type = None
    ckpt_hidden_size = None
    if isinstance(ckpt, dict):
        ckpt_model_type = ckpt.get("model_type", None)
        ckpt_hidden_size = ckpt.get("hidden_size", None)

    if ckpt_model_type is not None:
        model_type = ckpt_model_type
    if ckpt_hidden_size is not None:
        hidden_size = ckpt_hidden_size

    # Instantiate model
    OUTPUT_SIZE = len(CHARS) + 1
    model = CNN(
        input_chanels=1,
        output_channels=output_channels,
        input_size=repr_n_mels,
        hidden_size=hidden_size,
        n_classes=OUTPUT_SIZE,
        model_type=model_type,
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    # Load data
    wav_folder = Path(data_dir) / "wavs"
    metadata_file = Path(data_dir) / "metadata.csv"
    all_data = fill_rprs_from_folder(
        str(wav_folder),
        str(metadata_file),
        repr_type="mel",
        repr_n_mels=repr_n_mels,
        max_samples=max_samples,
        pad_sequences=False,
        skip_samples=0,
    )

    all_items = list(all_data.items())
    split_idx = int(len(all_items) * train_split)
    val_items = all_items[split_idx:]

    if len(val_items) == 0:
        print("No validation items, reduce train_split or increase dataset size")
        return

    val_data = {k: v for k, v in val_items}

    dataset = CNNDataset(val_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_cnn,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    total_char_err = 0
    total_chars = 0
    total_word_err = 0
    total_words = 0

    with torch.no_grad():
        for batch in dataloader:
            rpr = batch["rpr"].to(device)
            labels = batch["labels"].cpu()
            target_lengths = batch["target_lengths"].cpu()

            if rpr.dim() == 3:
                rpr = rpr.unsqueeze(1)

            preds = model(rpr)  # [T, batch, C]

            for i in range(labels.size(0)):
                # Reference
                ref_idx = labels[i, : target_lengths[i]].tolist()
                ref_chars = [idx_to_char.get(idx, "") for idx in ref_idx if idx != 0]
                ref_text = "".join(ref_chars)

                # Prediction
                pred_text = decode_prediction(preds[:, i : i + 1, :].cpu())

                cer_err, cer_len = compute_cer(pred_text, ref_text)
                wer_err, wer_len = compute_wer(pred_text, ref_text)

                total_char_err += cer_err
                total_chars += cer_len
                total_word_err += wer_err
                total_words += wer_len

    cer = 100.0 * total_char_err / total_chars if total_chars > 0 else 0.0
    wer = 100.0 * total_word_err / total_words if total_words > 0 else 0.0

    print("CNN Evaluation results:")
    print(f"  Samples evaluated: {len(val_data)}")
    print(f"  CER: {cer:.2f}%  ({total_char_err}/{total_chars})")
    print(f"  WER: {wer:.2f}%  ({total_word_err}/{total_words})")


def transformer_evaluate(
    checkpoint_path: str,
    data_dir: str,
    max_samples: int = None,
    device: str = "cuda",
    batch_size: int = 4,
    max_target_len: int = 200,
    train_split: float = 0.99,
    num_workers: int = 4,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    vectorizer = VectorizeChar(max_len=max_target_len)
    vocab_size = len(vectorizer.get_vocabulary())

    model = Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        source_maxlen=2754,
        target_maxlen=max_target_len,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=vocab_size,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Load data via transformer helper
    wav_folder = Path(data_dir) / "wavs"
    metadata_file = Path(data_dir) / "metadata.csv"

    data = get_data_from_folder(
        str(wav_folder), str(metadata_file), max_samples=max_samples
    )
    split_idx = int(len(data) * train_split)
    val_data = data[split_idx:]
    if len(val_data) == 0:
        print("No validation items, reduce train_split or increase dataset size")
        return

    val_paths = [d["audio"] for d in val_data]
    val_texts = [d["text"] for d in val_data]

    val_dataset = VectorizeCharDataset(val_paths, val_texts, vectorizer, pad_len=2754)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    total_char_err = 0
    total_chars = 0
    total_word_err = 0
    total_words = 0

    with torch.no_grad():
        for batch in val_loader:
            source = batch["source"].to(device)
            target = batch["target"].cpu()

            preds = model.generate(source, target_start_token_idx=2)
            preds = preds.cpu().numpy()

            for i in range(len(preds)):
                pred_indices = preds[i].tolist()
                pred_text = vectorizer.decode(pred_indices)

                ref_indices = target[i].numpy().tolist()
                ref_text = vectorizer.decode(ref_indices)

                cer_err, cer_len = compute_cer(pred_text, ref_text)
                wer_err, wer_len = compute_wer(pred_text, ref_text)

                total_char_err += cer_err
                total_chars += cer_len
                total_word_err += wer_err
                total_words += wer_len

    cer = 100.0 * total_char_err / total_chars if total_chars > 0 else 0.0
    wer = 100.0 * total_word_err / total_words if total_words > 0 else 0.0

    print("Transformer Evaluation results:")
    print(f"  Samples evaluated: {len(val_data)}")
    print(f"  CER: {cer:.2f}%  ({total_char_err}/{total_chars})")
    print(f"  WER: {wer:.2f}%  ({total_word_err}/{total_words})")


# A convenience dataset wrapper for transformer to reuse dataloader
class VectorizeCharDataset(torch.utils.data.Dataset):
    def __init__(
        self, audio_paths, texts, vectorizer: VectorizeChar, pad_len: int = 2754
    ):
        self.audio_paths = audio_paths
        self.texts = texts
        self.vectorizer = vectorizer
        self.pad_len = pad_len

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        from pipeline_transformer import path_to_audio

        audio = path_to_audio(self.audio_paths[idx], self.pad_len)
        text_indices = self.vectorizer(self.texts[idx])
        text = torch.tensor(text_indices, dtype=torch.long)
        return {"source": audio, "target": text}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute CER/WER for checkpoints")
    parser.add_argument("--model", choices=["cnn", "transformer", "mlp"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to LJSpeech folder")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--repr_n_mels", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--output_channels", type=int, default=64)
    parser.add_argument("--model_type", default="BI-LSTM")
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--num_workers", type=int, default=4)

    # Transformer-specific
    parser.add_argument("--max_target_len", type=int, default=200)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "cnn":
        cnn_evaluate(
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            device=args.device,
            batch_size=args.batch_size,
            repr_n_mels=args.repr_n_mels,
            hidden_size=args.hidden_size,
            model_type=args.model_type,
            output_channels=args.output_channels,
            train_split=args.train_split,
            num_workers=args.num_workers,
            checkpoint_path=args.checkpoint,
        )
    elif args.model == "mlp":
        mlp_evaluate(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            device=args.device,
            repr_n_mels=args.repr_n_mels if args.repr_n_mels is not None else 23,
            train_split=args.train_split,
        )
    elif args.model == "transformer":
        transformer_evaluate(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            device=args.device,
            batch_size=args.batch_size,
            train_split=args.train_split,
            num_workers=args.num_workers,
            max_target_len=args.max_target_len,
        )


def mlp_evaluate(
    checkpoint_path: str,
    data_dir: str,
    max_samples: int = None,
    device: str = "cuda",
    repr_n_mels: int = 23,
    train_split: float = 0.99,
):
    """Evaluate an MLP saved checkpoint using padded MFCC representations.
    We'll load a checkpoint (expected to include metadata hidden_size/n_layers) and compute CER/WER.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Read checkpoint and recreate model
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_hidden_size = None
    ckpt_n_layers = None
    if isinstance(ckpt, dict):
        ckpt_hidden_size = ckpt.get("hidden_size", None)
        ckpt_n_layers = ckpt.get("n_layers", None)
    hidden_size = ckpt_hidden_size if ckpt_hidden_size is not None else 32
    n_layers = ckpt_n_layers if ckpt_n_layers is not None else 2

    OUTPUT_SIZE = len(CHARS) + 1
    try:
        from MLP.model import MLP
    except Exception:
        from MLP.model import MLP

    model = MLP(
        input_size=repr_n_mels,
        hidden_size=hidden_size,
        output_size=OUTPUT_SIZE,
        n_layers=n_layers,
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()

    # Load padded data
    wav_folder = Path(data_dir) / "wavs"
    metadata_file = Path(data_dir) / "metadata.csv"
    all_data = fill_rprs_from_folder(
        str(wav_folder),
        str(metadata_file),
        repr_type="mfcc",
        repr_n_mels=repr_n_mels,
        max_samples=max_samples,
        pad_sequences=True,
        skip_samples=0,
    )

    all_items = list(all_data.items())
    split_idx = int(len(all_items) * train_split)
    val_items = all_items[split_idx:]

    if len(val_items) == 0:
        print("No validation items, reduce train_split or increase dataset size")
        return

    total_char_err = 0
    total_chars = 0
    total_word_err = 0
    total_words = 0

    with torch.no_grad():
        for name, sample in val_items:
            rpr = sample.rpr.to(device)
            label = sample.label
            # permute to [time, feats]
            input_to_mlp = rpr.permute(0, 2, 1).squeeze(0)
            preds = model(input_to_mlp)
            if preds.dim() == 2:
                preds = preds.unsqueeze(1)
            pred_text = decode_prediction(preds.cpu())
            # reference
            ref_idx = label.tolist() if label is not None else []
            ref_chars = [idx_to_char.get(idx, "") for idx in ref_idx if idx != 0]
            ref_text = "".join(ref_chars)
            cer_err, cer_len = compute_cer(pred_text, ref_text)
            wer_err, wer_len = compute_wer(pred_text, ref_text)
            total_char_err += cer_err
            total_chars += cer_len
            total_word_err += wer_err
            total_words += wer_len

    cer = 100.0 * total_char_err / total_chars if total_chars > 0 else 0.0
    wer = 100.0 * total_word_err / total_words if total_words > 0 else 0.0
    print("MLP Evaluation results:")
    print(f"  Samples evaluated: {len(val_items)}")
    print(f"  CER: {cer:.2f}%  ({total_char_err}/{total_chars})")
    print(f"  WER: {wer:.2f}%  ({total_word_err}/{total_words})")


if __name__ == "__main__":
    main()
