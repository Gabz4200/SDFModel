#!/usr/bin/env python3
"""
SDF Scene Reconstruction Training Script
Trains SDF models (CrossAttnSDFModel or VectorSDFModel) and learnable object embeddings to reconstruct a 4-primitive 3D scene.
"""

import argparse
import contextlib
import sys

import torch

from sdfmodel.datasets import build_scene_dataloader, build_voxel_dataloader
from sdfmodel.engine import SceneTrainer
from sdfmodel.models import CrossAttnSDFModel, VectorSDFModel
from sdfmodel.render import create_sdf3_wrapper, export_sdf_mesh
from sdfmodel.render_voxel import (
    create_voxel_model,
    export_voxel_obj,
    render_voxel_slice,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train SDF Model & Learnable Embeddings on 3D Scene"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default 2 for CPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-samples", type=int, default=512, help="Dataset sample count")
    parser.add_argument("--points-per-item", type=int, default=256, help="Points per item sample")
    parser.add_argument("--num-tokens", type=int, default=8, help="Learnable object token count")
    parser.add_argument(
        "--fourier-bands",
        type=int,
        default=8,
        help="Number of Fourier bands for cross-attn/vector models",
    )
    parser.add_argument(
        "--use-scene-token",
        action="store_true",
        help="Use scene-level token",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="scalar_sdf",
        choices=["scalar_sdf", "vector_sdf", "voxel"],
        help="Model type",
    )
    parser.add_argument(
        "--view",
        nargs="?",
        const="3d",
        default=None,
        choices=["2d", "3d"],
        help="Open live window showing SDF reconstruction",
    )
    parser.add_argument("--render-every-steps", type=int, default=5, help="Render update frequency")
    parser.add_argument("--render-resolution", type=int, default=128, help="Live render slice resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--save-checkpoint", type=str, default=None, help="Checkpoint path (.pt)")
    parser.add_argument("--save-mesh", type=str, default=None, help="Export mesh path (.stl/.obj)")
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    if args.model_type == "vector_sdf":
        model = VectorSDFModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_num_bands=args.fourier_bands,
            use_scene_token=args.use_scene_token,
        ).to(device)
    elif args.model_type == "voxel":
        from sdfmodel.models import build_model

        model = build_model(
            "cross_attn_voxel",
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_num_bands=args.fourier_bands,
            use_scene_token=args.use_scene_token,
        ).to(device)
    else:
        model = CrossAttnSDFModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_num_bands=args.fourier_bands,
            use_scene_token=args.use_scene_token,
        ).to(device)

    hidden_dim = args.hidden_dim
    embeddings = torch.nn.Parameter(torch.randn(args.num_tokens, hidden_dim, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [embeddings], lr=args.lr)

    if args.model_type == "voxel":
        dataloader = build_voxel_dataloader(
            voxel_path="data/donutvoxel.vox",
            num_samples=args.num_samples,
            points_per_item=args.points_per_item,
            batch_size=args.batch_size,
        )
    else:
        dataloader = build_scene_dataloader(
            num_samples=args.num_samples,
            points_per_item=args.points_per_item,
            batch_size=args.batch_size,
            return_normals=True,
        )

    total_steps = len(dataloader) * args.epochs

    print(f"Model parameters: {model.num_parameters:,} (trainable: {model.trainable_parameters:,})")
    print(f"Starting training on '{device}' for {args.epochs} epochs ({total_steps} steps)...")

    if args.model_type == "voxel":
        bce = torch.nn.BCELoss()
        mse = torch.nn.MSELoss()
        model.train()
        voxel_viewer = None
        render_every = max(1, args.render_every_steps)
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for step, (pts, targets) in enumerate(dataloader, start=1):
                pts = pts.to(device)
                targets = targets.to(device)
                coords = pts[..., :3]
                obj_emb = embeddings.unsqueeze(0).repeat(coords.shape[0], 1, 1)
                pred = model(coords, obj_emb)
                loss_exist = bce(pred[..., 0], targets[..., 0])
                loss_rgb = mse(pred[..., 1:], targets[..., 1:])
                loss = loss_exist + loss_rgb
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if (
                    args.view in ("2d", "3d")
                    and step % render_every == 0
                ):
                    voxel_model_eval = model
                    grid_shape = (
                        args.render_resolution,
                        args.render_resolution,
                        args.render_resolution,
                    )
                    vm = create_voxel_model(
                        voxel_model_eval,
                        embedding=embeddings,
                        device=device,
                        grid_shape=grid_shape,
                    )
                    import matplotlib.pyplot as plt

                    if voxel_viewer is not None:
                        with contextlib.suppress(Exception):
                            plt.close(voxel_viewer)
                    if args.view == "2d":
                        img = render_voxel_slice(
                            vm,
                            axis="z",
                            show=False,
                            title=(
                                f"Voxel Slice - Epoch {epoch+1} Step {step}"
                            ),
                        )
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(img)
                        ax.set_title(
                            f"Voxel Slice - Epoch {epoch+1} Step {step}"
                        )
                        ax.axis("off")
                        voxel_viewer = fig
                    else:
                        img_xy = render_voxel_slice(
                            vm, axis="z", show=False
                        )
                        img_xz = render_voxel_slice(
                            vm, axis="y", show=False
                        )
                        img_yz = render_voxel_slice(
                            vm, axis="x", show=False
                        )
                        fig, axes = plt.subplots(
                            1, 3, figsize=(12, 4)
                        )
                        axes[0].imshow(img_xy)
                        axes[0].set_title("XY slice")
                        axes[0].axis("off")
                        axes[1].imshow(img_xz)
                        axes[1].set_title("XZ slice")
                        axes[1].axis("off")
                        axes[2].imshow(img_yz)
                        axes[2].set_title("YZ slice")
                        axes[2].axis("off")
                        fig.suptitle(
                            f"Voxel 3D Slices - Epoch {epoch+1} Step {step}"
                        )
                        voxel_viewer = fig
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.01)

            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/len(dataloader):.6f}")
    else:
        trainer = SceneTrainer(
            model=model,
            learnable_embeddings=embeddings,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            view=args.view,
            render_every_steps=args.render_every_steps,
            render_resolution=args.render_resolution,
            model_type="scalar_sdf" if args.model_type == "scalar_sdf" else "vector_sdf",
            total_steps=total_steps,
            vector_warmup_steps=0,
            log_every_steps=10,
            w_distance=1.0,
            w_l1=0.5,
            w_eikonal=0.2,
            w_normal=0.5,
        )
        trainer.fit(epochs=args.epochs)

    if args.save_checkpoint:
        print(f"Saving checkpoint to '{args.save_checkpoint}'...")
        torch.save(
            {
                "model_state": model.state_dict(),
                "embedding_state": embeddings.detach().cpu(),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_tokens": args.num_tokens,
                "fourier_num_bands": args.fourier_bands,
                "use_scene_token": args.use_scene_token,
                "model_type": args.model_type,
            },
            args.save_checkpoint,
        )

    if args.save_mesh:
        if args.model_type == "voxel":
            print(f"Exporting voxel OBJ to '{args.save_mesh}'...")
            vm = create_voxel_model(
                model,
                embedding=embeddings.detach(),
                device=device,
            )
            export_voxel_obj(vm, args.save_mesh)
            print(f"Voxel OBJ saved to '{args.save_mesh}'!")
        else:
            print(f"Exporting final reconstructed 3D mesh to '{args.save_mesh}'...")
            sdf_obj = create_sdf3_wrapper(
                model, embedding=embeddings.detach(), device=device
            )
            export_sdf_mesh(sdf_obj, args.save_mesh, step=0.05, verbose=True)
            print(f"Final 3D mesh saved to '{args.save_mesh}'!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
