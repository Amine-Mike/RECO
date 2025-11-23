import argparse
import torch

from MLP.model import MLP
from CNN.model import CNN
from pipeline_mlp import MLPPipeline
from pipeline_cnn import CNNPipeline
from benchmarker import PipelineBenchmarker

OUTPUT_SIZE = len("abcdefghijklmnopqrstuvwxyz ") + 1


def main():
    parser = argparse.ArgumentParser(description="Run MLP or CNN pipeline")
    parser.add_argument("mode", choices=["mlp", "cnn"], help="Which pipeline to run")
    parser.add_argument("--device", default=None, help="CUDA device or cpu")
    # General pipeline flags
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to load for training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train (overrides defaults)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate to use (override)")
    parser.add_argument("--repr_n_mels", type=int, default=None, help="Number of mels for representations (override)")
    parser.add_argument("--save_checkpoints", action='store_true', help="Whether to save checkpoints during training")
    parser.add_argument("--checkpoint_dir", default=None, help="Where to write generated checkpoints")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.mode == "mlp":
        print(f"Model chosen MLP on device {device}")
        input_size = args.repr_n_mels if args.repr_n_mels is not None else 23
        hidden_size = 32
        n_layers = 2
        max_samples = args.max_samples if args.max_samples is not None else 100
        epochs = args.epochs if args.epochs is not None else 20
        lr = args.lr if args.lr is not None else 3e-4
        repr_n_mels = input_size
        checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else "checkpoints_mlp"
        save_checkpoints = args.save_checkpoints
        mlp_model = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=OUTPUT_SIZE,
            n_layers=n_layers,
        )
        pipeline = MLPPipeline(
            mlp_model,
            repr_type="mfcc",
            repr_n_mels=repr_n_mels,
            device=device,
            max_samples=max_samples,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=save_checkpoints,
            hidden_size=hidden_size,
            n_layers=n_layers,
        )
        print("Starting MLP pipeline...")
        pipeline.launch_pipeline(epochs=epochs, lr=lr, checkpoint_dir=checkpoint_dir)
    else:
        print(f"Model chosen CNN on device {device}")
        input_channels = 1
        output_channels = 64
        input_size = 128
        hidden_size = 128
        max_samples = 2000
        model_type = "BI-LSTM"  # Options: "LSTM", "BI-LSTM", "GRU"
        print(f"Preparing CNN model... with {model_type} predicition head")
        model = CNN(
            input_chanels=input_channels,
            output_channels=output_channels,
            input_size=input_size,
            hidden_size=hidden_size,
            n_classes=OUTPUT_SIZE,
            model_type=model_type,
        )
        pipeline = CNNPipeline(
            model,
            repr_type="mel",
            repr_n_mels=input_size,
            device=device,
            max_samples=max_samples,
        )
        print("Starting CNN pipeline...")
        benchmark = PipelineBenchmarker(pipeline.launch_pipeline)
        benchmark.time_execution()


if __name__ == "__main__":
    main()
