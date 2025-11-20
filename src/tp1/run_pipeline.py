import argparse
import torch

from MLP.model import MLP
from CNN.model import CNN
from pipeline_mlp import MLPPipeline
from pipeline_cnn import CNNPipeline


def main():
    parser = argparse.ArgumentParser(description="Run MLP or CNN pipeline")
    parser.add_argument("mode", choices=["mlp", "cnn"], help="Which pipeline to run")
    parser.add_argument("--device", default=None, help="CUDA device or cpu")
    args = parser.parse_args()

    device = (
        torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.mode == "mlp":
        print(f"Model chosen MLP on device {device}")
        INPUT_SIZE = 23
        HIDDEN_SIZE = 32
        OUTPUT_SIZE = len("abcdefghijklmnopqrstuvwxyz ") + 1
        N_LAYERS = 2
        mlp_model = MLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, n_layers=N_LAYERS)
        pipeline = MLPPipeline(mlp_model, repr_type="mfcc", repr_n_mels=INPUT_SIZE, device=device, max_samples=100)
        print("Starting MLP pipeline...")
        pipeline.launch_pipeline()
    else:
        print(f"Model chosen CNN on device {device}")
        INPUT_SIZE = 128
        HIDDEN_SIZE = 128
        OUTPUT_SIZE = len("abcdefghijklmnopqrstuvwxyz ") + 1
        model = CNN(input_chanels=1, output_channels=64, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, n_classes=OUTPUT_SIZE)
        pipeline = CNNPipeline(model, repr_type="mel", repr_n_mels=INPUT_SIZE, device=device, max_samples=100)
        print("Starting CNN pipeline...")
        pipeline.launch_pipeline()


if __name__ == "__main__":
    main()
