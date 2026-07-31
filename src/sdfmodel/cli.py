import argparse
import sys

import torch

from sdfmodel.datasets import build_dataloaders
from sdfmodel.engine import Trainer
from sdfmodel.models import CrossAttnSDFModel, build_model, list_models
from sdfmodel.render import (
    create_sdf3_wrapper,
    export_sdf_mesh,
    render_sdf_3d,
    render_sdf_slice,
)
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


def run_render(args: argparse.Namespace) -> int:
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"Initializing model '{args.model}' on device '{device}'...")

    if args.model == "cross_attn_sdf":
        model = CrossAttnSDFModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        model = build_model(
            args.model,
            hidden_features=args.hidden_dim,
            num_layers=args.num_layers,
        )

    if args.checkpoint:
        print(f"Loading weights from '{args.checkpoint}'...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device).eval()

    embedding = None
    if isinstance(model, CrossAttnSDFModel):
        embedding = CrossAttnSDFModel.create_learnable_embedding(
            batch_size=1,
            seq_len=4,
            hidden_dim=args.hidden_dim,
            device=torch.device(device),
        )

    print("Wrapping model in fogleman/sdf SDF3 interface...")
    sdf_obj = create_sdf3_wrapper(model, embedding=embedding, device=device)

    x_val = args.slice_pos if args.slice_axis == "x" else None
    y_val = args.slice_pos if args.slice_axis == "y" else None
    z_val = args.slice_pos if args.slice_axis == "z" else None

    if args.mode == "slice":
        print(
            f"Rendering 2D SDF slice along {args.slice_axis.upper()}={args.slice_pos} at {args.resolution}x{args.resolution} resolution..."
        )
        render_sdf_slice(
            sdf_obj,
            resolution=args.resolution,
            x=x_val,
            y=y_val,
            z=z_val,
            show=True,
            title=f"SDF Slice ({args.model}, {args.slice_axis.upper()}={args.slice_pos})",
        )

    elif args.mode in ("3d", "mesh"):
        print(f"Extracting 3D isosurface mesh with step size {args.step}...")
        triangles = render_sdf_3d(
            sdf_obj,
            step=args.step,
            show=(args.mode == "3d"),
            title=f"3D SDF Isosurface Mesh ({args.model})",
        )
        print(f"Generated {len(triangles)} triangles in 3D mesh.")

    if args.output_mesh:
        print(f"Exporting 3D mesh to '{args.output_mesh}'...")
        export_sdf_mesh(
            sdf_obj,
            args.output_mesh,
            step=args.step,
            verbose=True,
        )
        print(f"Mesh successfully saved to '{args.output_mesh}'!")

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

    # Render command
    render_parser = subparsers.add_parser(
        "render", help="Render SDF model slice or 3D mesh using fogleman/sdf"
    )
    render_parser.add_argument(
        "--model",
        type=str,
        default="cross_attn_sdf",
        choices=list_models(),
        help="SDF model variant to evaluate",
    )
    render_parser.add_argument(
        "--mode",
        type=str,
        default="slice",
        choices=["slice", "3d", "mesh"],
        help="Rendering mode: 2D slice, 3D interactive mesh view, or mesh export",
    )
    render_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size for model",
    )
    render_parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers for model",
    )
    render_parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Grid resolution for 2D slice sampling",
    )
    render_parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Marching cubes step size for 3D mesh generation",
    )
    render_parser.add_argument(
        "--slice-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Axis for 2D slice plane",
    )
    render_parser.add_argument(
        "--slice-pos",
        type=float,
        default=0.0,
        help="Position coordinate along slice-axis",
    )
    render_parser.add_argument(
        "--output-mesh",
        type=str,
        default=None,
        help="File path to save output 3D mesh (.stl or .obj)",
    )
    render_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (cpu, cuda, auto)",
    )
    render_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch model checkpoint (.pt or .pth)",
    )

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
    elif args.command == "render":
        return run_render(args)
    else:
        cfg = ExperimentConfig()
        return run_train(cfg)


if __name__ == "__main__":
    sys.exit(main())
