import argparse
import sys

import torch

from sdfmodel.datasets import build_dataloaders
from sdfmodel.engine import Trainer
from sdfmodel.models import build_model, list_models
from sdfmodel.utils import ExperimentConfig, seed_everything


def run_train(config: ExperimentConfig) -> int:
    seed_everything(config.training.seed)
    print(f"Building model '{config.model.name}'...")
    model = build_model(
        config.model.name,
        in_features=config.model.in_features,
        hidden_features=config.model.hidden_features,
        num_layers=config.model.num_layers,
        out_features=config.model.out_features,
        use_fourier_pe=config.model.use_fourier_pe,
        fourier_num_bands=config.model.fourier_num_bands,
    )
    print(
        f"Model parameters: {model.num_parameters:,} (trainable: {model.trainable_parameters:,})"
    )

    print(f"Building dataset '{config.dataset.name}'...")
    train_loader, val_loader = build_dataloaders(
        config.dataset, seed=config.training.seed
    )

    print("Starting training loop...")
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, config=config
    )
    final_metrics = trainer.fit()

    print(
        f"Training completed! Final Val Metrics: MSE={final_metrics['mse']:.6f}, PSNR={final_metrics['psnr']:.2f} dB"
    )
    return 0


def run_info() -> int:
    print("Registered SDF Models:", list_models())
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SDFModel PyTorch Research CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu, cuda, auto)"
    )

    # Info command
    subparsers.add_parser("info", help="Display environment and model info")

    args = parser.parse_args()

    if args.command == "train":
        cfg = ExperimentConfig()
        cfg.training.epochs = args.epochs
        cfg.dataset.batch_size = args.batch_size
        cfg.training.learning_rate = args.lr
        cfg.training.seed = args.seed
        cfg.training.device = args.device
        return run_train(cfg)
    elif args.command == "info":
        return run_info()
    else:
        cfg = ExperimentConfig()
        return run_train(cfg)


if __name__ == "__main__":
    sys.exit(main())
