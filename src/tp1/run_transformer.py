"""
Main script to train the Transformer model for speech recognition.

Usage:
    python run_transformer.py --data_dir /path/to/LJSpeech-1.1 --epochs 100 --batch_size 64
"""

import argparse
from pathlib import Path

import torch
from pipeline_transformer import (
    VectorizeChar,
    create_dataloaders,
    get_data_from_folder,
    train_transformer,
)
from TRANSFORMER.model import Transformer


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer speech recognition model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../LJSpeech-1.1",
        help="Path to LJSpeech dataset directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)",
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=200,
        help="Maximum length of target sequences",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_hid", type=int, default=200, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_head", type=int, default=2, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_feed_forward", type=int, default=400, help="Feed-forward dimension"
    )
    parser.add_argument(
        "--num_layers_enc", type=int, default=4, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_layers_dec", type=int, default=1, help="Number of decoder layers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_transformer",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--display_every",
        type=int,
        default=5,
        help="Display predictions every N epochs",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Transformer Speech Recognition Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden dim: {args.num_hid}")
    print(f"Attention heads: {args.num_head}")
    print(f"Encoder layers: {args.num_layers_enc}")
    print(f"Decoder layers: {args.num_layers_dec}")
    print("=" * 80)

    # Setup paths
    data_dir = Path(args.data_dir)
    wav_folder = data_dir / "wavs"
    metadata_path = data_dir / "metadata.csv"

    if not wav_folder.exists():
        raise FileNotFoundError(f"Wav folder not found: {wav_folder}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Initialize vectorizer
    print("\nInitializing vectorizer...")
    vectorizer = VectorizeChar(max_len=args.max_target_len)
    vocab_size = len(vectorizer.get_vocabulary())
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {vectorizer.get_vocabulary()}")

    # Load data
    print("\nLoading data...")
    data = get_data_from_folder(
        str(wav_folder), str(metadata_path), max_samples=args.max_samples
    )
    print(f"Total samples: {len(data)}")

    if len(data) == 0:
        raise ValueError("No data loaded! Check your data directory and metadata file.")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data=data,
        vectorizer=vectorizer,
        batch_size=args.batch_size,
        train_split=0.99,
        pad_len=2754,
        num_workers=args.num_workers,
    )

    # Create model
    print("\nCreating Transformer model...")
    model = Transformer(
        num_hid=args.num_hid,
        num_head=args.num_head,
        num_feed_forward=args.num_feed_forward,
        source_maxlen=2754,  # Fixed audio length
        target_maxlen=args.max_target_len,
        num_layers_enc=args.num_layers_enc,
        num_layers_dec=args.num_layers_dec,
        num_classes=vocab_size,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\nStarting training...")
    print("=" * 80)
    train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vectorizer=vectorizer,
        epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        display_every=args.display_every,
    )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
