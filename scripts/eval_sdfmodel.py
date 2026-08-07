#!/usr/bin/env python3
"""
SDF Model Interactive Evaluation & Primitive Embedding Interpolation Script
Loads a trained CrossAttnSDFModel checkpoint and opens an interactive window with sliders
to interpolate primitive embeddings in real time.
"""

import argparse
import sys

import torch

from sdfmodel.models import CrossAttnSDFModel
from sdfmodel.render import render_interactive_interpolation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive Evaluation & Embedding Interpolation for CrossAttnSDFModel"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to PyTorch model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--view",
        nargs="?",
        const="3d",
        default="3d",
        choices=["2d", "3d"],
        help="Visualization view mode: '2d' slice or '3d' isosurface mesh (defaults to '3d')",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.10,
        help="Marching cubes step size for 3D rendering",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Grid resolution for 2D slice sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (cpu, cuda, auto)",
    )

    args = parser.parse_args()

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
        from sdfmodel.models import VectorSDFModel

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
            batch_size=1,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            device=torch.device(device),
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


if __name__ == "__main__":
    sys.exit(main())
