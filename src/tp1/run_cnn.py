"""
Main script to train the CNN model for speech recognition with DataLoader.

Usage:
    python run_cnn.py --data_dir data/LJSpeech-1.1 --epochs 30 --batch_size 32 --max_samples 2000
"""

import argparse
from pathlib import Path

import torch
from benchmarker import PipelineBenchmarker
from CNN.model import CNN
from pipeline_cnn import CNNPipeline
from pipeline_common import CHARS


def main():
    parser = argparse.ArgumentParser(description="Train CNN speech recognition model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/LJSpeech-1.1",
        help="Path to LJSpeech dataset directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of samples to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--repr_n_mels", type=int, default=128, help="Number of mel bins"
    )
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument(
        "--output_channels", type=int, default=64, help="CNN output channels"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BI-LSTM",
        help="Model that will be plugged to the Convolutions",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CNN+LSTM Speech Recognition Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Mel bins: {args.repr_n_mels}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"CNN + {args.model}")
    print("=" * 80)

    data_dir = Path(args.data_dir)
    wav_folder = data_dir / "wavs"
    metadata_path = data_dir / "metadata.csv"

    if not wav_folder.exists():
        raise FileNotFoundError(f"Wav folder not found: {wav_folder}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    OUTPUT_SIZE = len(CHARS) + 1
    model = CNN(
        input_chanels=1,
        output_channels=args.output_channels,
        input_size=args.repr_n_mels,
        hidden_size=args.hidden_size,
        n_classes=OUTPUT_SIZE,
        model_type=args.model,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    pipeline = CNNPipeline(
        model=model,
        folder=str(wav_folder),
        metadata=str(metadata_path),
        repr_type="mel",
        repr_n_mels=args.repr_n_mels,
        device=torch.device(args.device),
        max_samples=args.max_samples,
    )

    print("\nLoading data...")
    pipeline.fill_rprs()

    print("\nStarting training...")
    print("=" * 80)

    def training_fn():
        return pipeline.train_model(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    benchmarker = PipelineBenchmarker(training_fn)
    benchmarker.time_execution()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
