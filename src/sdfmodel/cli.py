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
            fourier_num_bands=args.fourier_bands,
            use_scene_token=args.use_scene_token,
        )
    elif args.model == "vector_sdf":
        model = VectorSDFModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_num_bands=args.fourier_bands,
            use_scene_token=args.use_scene_token,
        )
    else:
        model = build_model(
            args.model,
            hidden_features=args.hidden_dim,
            num_layers=args.num_layers,
        )

    num_tokens = args.num_tokens
    checkpoint = None
    if args.checkpoint:
        print(f"Loading weights from '{args.checkpoint}'...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(model, (CrossAttnSDFModel, VectorSDFModel)):
            # Checkpoint architecture flags are authoritative: rebuild when any
            # of them differs from the CLI args (band count, layers, hidden dim,
            # and scene token all change state_dict shapes).
            ckpt_hidden = checkpoint.get("hidden_dim", args.hidden_dim)
            ckpt_layers = checkpoint.get("num_layers", args.num_layers)
            ckpt_bands = checkpoint.get("fourier_num_bands", args.fourier_bands)
            ckpt_token = checkpoint.get("use_scene_token", args.use_scene_token)
            if (ckpt_hidden, ckpt_layers, ckpt_bands, ckpt_token) != (
                args.hidden_dim,
                args.num_layers,
                args.fourier_bands,
                args.use_scene_token,
            ):
                if isinstance(model, VectorSDFModel):
                    model = VectorSDFModel(
                        hidden_dim=ckpt_hidden,
                        num_layers=ckpt_layers,
                        fourier_num_bands=ckpt_bands,
                        use_scene_token=ckpt_token,
                    )
                else:
                    model = CrossAttnSDFModel(
                        hidden_dim=ckpt_hidden,
                        num_layers=ckpt_layers,
                        fourier_num_bands=ckpt_bands,
                        use_scene_token=ckpt_token,
                    )
            num_tokens = checkpoint.get("num_tokens", num_tokens)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))

    model = model.to(device).eval()

    embedding = None
    if isinstance(model, (CrossAttnSDFModel, VectorSDFModel)):
        embedding = CrossAttnSDFModel.create_learnable_embedding(
            batch_size=1,
            seq_len=num_tokens,
            hidden_dim=model.hidden_dim,
            device=torch.device(device),
        )
        if checkpoint is not None and "embedding_state" in checkpoint:
            embedding.data.copy_(checkpoint["embedding_state"].to(device))

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

    num_tokens = getattr(args, "num_tokens", 8)
    fourier_bands = getattr(args, "fourier_bands", 8)
    use_scene_token = getattr(args, "use_scene_token", False)
    surface_eps = getattr(args, "surface_eps", 0.1)
    sampler = getattr(args, "sampler", "chaos_game")
    chaos_iters = getattr(args, "chaos_iters", 4)
    vector_warmup = getattr(args, "vector_warmup", 0)
    log_every = getattr(args, "log_every", 10)

    if model_type == "vector_sdf":
        print(
            f"Initializing VectorSDFModel (hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, {num_tokens} tokens) and {num_tokens} learnable embeddings..."
        )
        model = VectorSDFModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_num_bands=fourier_bands,
            use_scene_token=use_scene_token,
        ).to(device)
    else:
        print(
            f"Initializing CrossAttnSDFModel (hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, {num_tokens} tokens) and {num_tokens} learnable embeddings..."
        )
        model = CrossAttnSDFModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_num_bands=fourier_bands,
            use_scene_token=use_scene_token,
        ).to(device)

    embeddings = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1,
        seq_len=num_tokens,
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
        surface_eps=surface_eps,
        sampler=sampler,
        chaos_iters=chaos_iters,
    )

    total_steps = len(dataloader) * args.epochs
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
        total_steps=total_steps,
        vector_warmup_steps=vector_warmup,
        log_every_steps=log_every,
        # Tuned weights: balance distance against eikonal + normal consistency.
        w_distance=1.0,
        w_l1=0.5,
        w_eikonal=0.2,
        w_normal=0.5,
        w_vector_l2=1.0,
        w_cosine=0.8,
        w_magnitude_mse=0.5,
        w_consistency=0.3,
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
                "num_tokens": num_tokens,
                "fourier_num_bands": fourier_bands,
                "use_scene_token": use_scene_token,
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
    fourier_num_bands = checkpoint.get("fourier_num_bands", 6)
    use_scene_token = checkpoint.get("use_scene_token", False)

    if model_type == "vector_sdf":
        print(
            f"Initializing VectorSDFModel (hidden_dim={hidden_dim}, num_layers={num_layers}, fourier_bands={fourier_num_bands}, scene_token={use_scene_token}) on device '{device}'..."
        )
        model = VectorSDFModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            fourier_num_bands=fourier_num_bands,
            use_scene_token=use_scene_token,
        ).to(device)
    else:
        print(
            f"Initializing CrossAttnSDFModel (hidden_dim={hidden_dim}, num_layers={num_layers}, fourier_bands={fourier_num_bands}, scene_token={use_scene_token}) on device '{device}'..."
        )
        model = CrossAttnSDFModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            fourier_num_bands=fourier_num_bands,
            use_scene_token=use_scene_token,
        ).to(device)

    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state)
    model.eval()

    if "embedding_state" in checkpoint:
        embedding = checkpoint["embedding_state"].to(device)
    else:
        print(
            "Warning: No 'embedding_state' found in checkpoint; initializing default embeddings..."
        )
        seq_len = checkpoint.get("num_tokens", 4)
        embedding = CrossAttnSDFModel.create_learnable_embedding(
            batch_size=1, seq_len=seq_len, hidden_dim=hidden_dim, device=torch.device(device)
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
    render_parser.add_argument(
        "--fourier-bands",
        type=int,
        default=6,
        help="Number of Fourier bands for cross-attn/vector models",
    )
    render_parser.add_argument(
        "--num-tokens",
        type=int,
        default=8,
        help="Learnable object token count for cross-attn/vector models",
    )
    render_parser.add_argument(
        "--use-scene-token",
        action="store_true",
        help="Use a learnable scene summary token (must match checkpoint)",
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
    scene_parser.add_argument(
        "--num-tokens",
        type=int,
        default=8,
        help="Learnable object token count (default 8: 4 primitives x 2 slots)",
    )
    scene_parser.add_argument(
        "--surface-eps",
        type=float,
        default=0.1,
        help="Surface-band sampling threshold for near-surface points",
    )
    scene_parser.add_argument(
        "--sampler",
        choices=("chaos_game", "rejection"),
        default="chaos_game",
        help="Near-surface sampling strategy: chaos_game (iterative surface projection, "
        "default) or rejection (uniform rejection inside surface_eps)",
    )
    scene_parser.add_argument(
        "--chaos-iters",
        type=int,
        default=4,
        help="Chaos-game warm-up iterations (project + jitter rounds)",
    )
    scene_parser.add_argument(
        "--fourier-bands",
        type=int,
        default=8,
        help="Number of Fourier bands (annealed from 4 up to this value)",
    )
    scene_parser.add_argument(
        "--use-scene-token",
        action="store_true",
        help="Add a learnable scene summary token coords attend to",
    )
    scene_parser.add_argument(
        "--vector-warmup",
        type=int,
        default=0,
        help="Scalar-only warmup steps before enabling vector losses",
    )
    scene_parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print per-term loss breakdown every N steps",
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
