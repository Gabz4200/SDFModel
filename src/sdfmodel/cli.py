import argparse
import sys


import torch

from sdfmodel.datasets import build_dataloaders, build_scene_dataloader
from sdfmodel.engine import SceneTrainer, Trainer
from sdfmodel.models import CrossAttnSDFModel, VectorSDFModel, build_model, list_models
from sdfmodel.render import (
    create_sdf3_wrapper,
    export_sdf_mesh,
    render_interactive_interpolation,
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
    elif args.model == "vector_sdf":
        model = VectorSDFModel(
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
    if isinstance(model, (CrossAttnSDFModel, VectorSDFModel)):
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

    view_mode = args.view
    if view_mode is None:
        view_mode = "3d"

    if view_mode == "2d":
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

    else:
        print(f"Extracting 3D isosurface mesh with step size {args.step}...")
        triangles = render_sdf_3d(
            sdf_obj,
            step=args.step,
            show=True,
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


def run_train_scene(args: argparse.Namespace) -> int:
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    model_type = getattr(args, "model_type", "scalar_sdf")
    assert model_type in ("scalar_sdf", "vector_sdf")

    if model_type == "vector_sdf":
        print(
            f"Initializing VectorSDFModel (hidden_dim={args.hidden_dim}, num_layers={args.num_layers}) and 4 learnable embeddings..."
        )
        model = VectorSDFModel(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers
        ).to(device)
    else:
        print(
            f"Initializing CrossAttnSDFModel (hidden_dim={args.hidden_dim}, num_layers={args.num_layers}) and 4 learnable embeddings..."
        )
        model = CrossAttnSDFModel(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers
        ).to(device)

    embeddings = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1,
        seq_len=4,
        hidden_dim=args.hidden_dim,
        device=torch.device(device),
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [embeddings], lr=args.lr)

    print(
        f"Building 4-primitive scene dataset ({args.num_samples} samples, batch_size={args.batch_size})..."
    )
    dataloader = build_scene_dataloader(
        num_samples=args.num_samples,
        points_per_item=args.points_per_item,
        batch_size=args.batch_size,
        return_normals=True,
    )

    trainer = SceneTrainer(
        model=model,
        learnable_embeddings=embeddings,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        view=args.view,
        render_every_steps=args.render_every_steps,
        render_resolution=args.render_resolution,
        model_type=model_type,
    )

    print("Starting 4-primitive scene training loop...")
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                points = batch[0]
                targets = batch[1]
                target_normals = batch[2] if len(batch) > 2 else None
                loss_val = trainer.train_step(
                    global_step, points, targets, target_normals=target_normals
                )
                epoch_loss += loss_val
                num_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(1, num_batches)
            print(f"Epoch [{epoch}/{args.epochs}] | Avg Loss: {avg_loss:.6f}")

    finally:
        trainer.close(keep_open=args.view is not None)

    print("Scene training finished!")

    if args.save_checkpoint:
        print(f"Saving checkpoint to '{args.save_checkpoint}'...")
        torch.save(
            {
                "model_state": model.state_dict(),
                "embedding_state": embeddings.data,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "model_type": model_type,
            },
            args.save_checkpoint,
        )

    if args.save_mesh:
        print(f"Exporting final reconstructed 3D mesh to '{args.save_mesh}'...")
        sdf_obj = create_sdf3_wrapper(
            model, embedding=embeddings.detach(), device=device
        )
        export_sdf_mesh(sdf_obj, args.save_mesh, step=0.05, verbose=True)
        print(f"Final 3D mesh saved to '{args.save_mesh}'!")

    return 0


def run_eval_sdfmodel(args: argparse.Namespace) -> int:
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    view_mode = args.view if args.view in ("2d", "3d") else "3d"

    print(f"Loading checkpoint from '{args.checkpoint}'...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    hidden_dim = checkpoint.get("hidden_dim", 64)
    num_layers = checkpoint.get("num_layers", 4)
    model_type = checkpoint.get("model_type", "scalar_sdf")

    if model_type == "vector_sdf":
        print(
            f"Initializing VectorSDFModel (hidden_dim={hidden_dim}, num_layers={num_layers}) on device '{device}'..."
        )
        model = VectorSDFModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    else:
        print(
            f"Initializing CrossAttnSDFModel (hidden_dim={hidden_dim}, num_layers={num_layers}) on device '{device}'..."
        )
        model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state)
    model.eval()

    if "embedding_state" in checkpoint:
        embedding = checkpoint["embedding_state"].to(device)
    else:
        print(
            "Warning: No 'embedding_state' found in checkpoint; initializing default embeddings..."
        )
        embedding = CrossAttnSDFModel.create_learnable_embedding(
            batch_size=1, seq_len=4, hidden_dim=hidden_dim, device=torch.device(device)
        )

    print(
        f"Launching interactive GUI window with sliders in '{view_mode.upper()}' view mode..."
    )
    render_interactive_interpolation(
        model=model,
        embeddings=embedding,
        step=args.step,
        resolution=args.resolution,
        view_mode=view_mode,
        device=device,
        title="Interactive Scene Primitive Embedding Interpolation",
    )

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
    render_parser.add_argument(
        "--view",
        nargs="?",
        const="3d",
        default=None,
        choices=["2d", "3d"],
        help="Visualization view mode: '2d' slice or '3d' isosurface mesh (defaults to '3d' if --view is specified without value)",
    )

    # Train scene command
    scene_parser = subparsers.add_parser(
        "train-scene", help="Train SDF model & 4 embeddings on 3D scene"
    )
    scene_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    scene_parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default 2 for CPU)"
    )
    scene_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    scene_parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Model hidden dimension"
    )
    scene_parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    scene_parser.add_argument(
        "--num-samples", type=int, default=512, help="Dataset sample count"
    )
    scene_parser.add_argument(
        "--points-per-item", type=int, default=256, help="Points per item sample"
    )
    scene_parser.add_argument(
        "--model-type",
        type=str,
        default="scalar_sdf",
        choices=["scalar_sdf", "vector_sdf"],
        help="Model type: 'scalar_sdf' (CrossAttnSDFModel) or 'vector_sdf' (VectorSDFModel)",
    )
    scene_parser.add_argument(
        "--view",
        nargs="?",
        const="3d",
        default=None,
        choices=["2d", "3d"],
        help="Open live window showing SDF reconstruction updating during training ('2d' slice or '3d' mesh, defaults to '3d')",
    )
    scene_parser.add_argument(
        "--render-every-steps",
        type=int,
        default=5,
        help="Render update frequency in steps (default 5)",
    )
    scene_parser.add_argument(
        "--render-resolution",
        type=int,
        default=128,
        help="Live render slice resolution",
    )
    scene_parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu, cuda, auto)"
    )
    scene_parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Path to save output PyTorch checkpoint (.pt)",
    )
    scene_parser.add_argument(
        "--save-mesh",
        type=str,
        default=None,
        help="Path to export final reconstructed 3D mesh (.stl)",
    )
    # Eval SDF Model command
    eval_parser = subparsers.add_parser(
        "eval-sdfmodel",
        help="Open interactive GUI window with sliders to interpolate scene primitive embeddings",
    )
    eval_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to PyTorch model checkpoint (.pt file)",
    )
    eval_parser.add_argument(
        "--view",
        nargs="?",
        const="3d",
        default="3d",
        choices=["2d", "3d"],
        help="Visualization view mode: '2d' slice or '3d' isosurface mesh (defaults to '3d')",
    )
    eval_parser.add_argument(
        "--step",
        type=float,
        default=0.10,
        help="Marching cubes step size for 3D rendering",
    )
    eval_parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Grid resolution for 2D slice sampling",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (cpu, cuda, auto)",
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
    elif args.command == "train-scene":
        return run_train_scene(args)
    elif args.command == "eval-sdfmodel":
        return run_eval_sdfmodel(args)
    else:
        cfg = ExperimentConfig()
        return run_train(cfg)


if __name__ == "__main__":
    sys.exit(main())
