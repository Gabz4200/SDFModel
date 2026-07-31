#!/usr/bin/env python3
"""
SDF Model Inference and Rendering Script
Evaluates PyTorch Signed Distance Field models and displays rendered outputs using fogleman/sdf.
"""

import argparse
import sys

import torch

from sdfmodel.models import CrossAttnSDFModel, build_model, list_models
from sdfmodel.render import (
    create_sdf3_wrapper,
    export_sdf_mesh,
    render_sdf_3d,
    render_sdf_slice,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SDF Model Inference & Rendering with fogleman/sdf"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cross_attn_sdf",
        choices=list_models(),
        help="SDF model variant to evaluate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="slice",
        choices=["slice", "3d", "mesh"],
        help="Rendering mode: 2D slice, 3D interactive mesh view, or mesh export",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size for model",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers for model",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Grid resolution for 2D slice sampling",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Marching cubes step size for 3D mesh generation",
    )
    parser.add_argument(
        "--slice-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Axis for 2D slice plane",
    )
    parser.add_argument(
        "--slice-pos",
        type=float,
        default=0.0,
        help="Position coordinate along slice-axis",
    )
    parser.add_argument(
        "--output-mesh",
        type=str,
        default=None,
        help="File path to save output 3D mesh (.stl or .obj)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch model checkpoint (.pt or .pth)",
    )

    args = parser.parse_args()

    # Determine computing device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Initializing model '{args.model}' on device '{device}'...")

    if args.model == "cross_attn_sdf":
        model: torch.nn.Module = CrossAttnSDFModel(
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

    # Create learnable embedding if model is CrossAttnSDFModel
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


if __name__ == "__main__":
    sys.exit(main())
