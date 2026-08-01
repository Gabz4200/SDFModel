#!/usr/bin/env python3
"""
SDF Scene Reconstruction Training Script
Trains CrossAttnSDFModel and 4 learnable primitive object embeddings to reconstruct a 4-primitive 3D scene.
Includes --view flag for non-blocking live rendering updates during training.
"""

import argparse
import sys

import torch

from sdfmodel.datasets import build_scene_dataloader
from sdfmodel.engine import SceneTrainer
from sdfmodel.models import CrossAttnSDFModel
from sdfmodel.render import create_sdf3_wrapper, export_sdf_mesh


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CrossAttnSDFModel & 4 Learnable Embeddings on 3D Scene"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default 2 for CPU)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Model hidden dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-samples", type=int, default=512, help="Dataset sample count"
    )
    parser.add_argument(
        "--points-per-item", type=int, default=256, help="Points per item sample"
    )
    parser.add_argument(
        "--view",
        nargs="?",
        const="3d",
        default=None,
        choices=["2d", "3d"],
        help="Open live window showing SDF reconstruction updating during training ('2d' slice or '3d' mesh, defaults to '3d')",
    )
    parser.add_argument(
        "--render-every-steps",
        type=int,
        default=5,
        help="Render update frequency in steps",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=128,
        help="Live render slice resolution",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Path to save output PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--save-mesh",
        type=str,
        default=None,
        help="Path to export final reconstructed 3D mesh (.stl)",
    )

    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

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
        f"Building scene dataset ({args.num_samples} samples, batch_size={args.batch_size})..."
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
    )

    print("Starting scene reconstruction training loop...")
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                points, targets = batch[0], batch[1]
                target_normals = batch[2] if len(batch) > 2 else None
                loss_val = trainer.train_step(
                    global_step, points, targets, target_normals=target_normals
                )
                epoch_loss += loss_val
                num_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(1, num_batches)
            print(f"Epoch [{epoch}/{args.epochs}] | Avg MSE Loss: {avg_loss:.6f}")

    finally:
        trainer.close(keep_open=args.view is not None)

    print("Training finished!")

    if args.save_checkpoint:
        print(f"Saving checkpoint to '{args.save_checkpoint}'...")
        torch.save(
            {
                "model_state": model.state_dict(),
                "embedding_state": embeddings.data,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
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


if __name__ == "__main__":
    sys.exit(main())
