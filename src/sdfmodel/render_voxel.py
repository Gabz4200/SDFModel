from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from voxelmap import Model

from sdfmodel.models.base import BaseModel


def create_voxel_model(
    model: BaseModel,
    embedding: torch.Tensor | None = None,
    device: str = "cpu",
    grid_shape: tuple[int, int, int] | None = None,
    voxel_origin: tuple[float, float, float] | None = None,
    batch_size: int = 65536,
) -> Model:
    """Wrap a voxel prediction model into a voxelmap Model for rendering and export.

    The model must output a 4D vector per coordinate: (exist, red, green, blue).
    The model is evaluated on the full grid, and the resulting array is stored
    in the returned Model instance.

    If grid_shape/voxel_origin are not provided, a unit cube centered at the
    origin is assumed.
    """
    model = model.to(device).eval()
    if embedding is not None:
        embedding = embedding.to(device)
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(0)

    if grid_shape is None:
        grid_shape = (32, 32, 32)
    if voxel_origin is None:
        voxel_origin = (-grid_shape[0] / 2.0, -grid_shape[1] / 2.0, -grid_shape[2] / 2.0)

    zz, yy, xx = np.mgrid[
        0 : grid_shape[0],
        0 : grid_shape[1],
        0 : grid_shape[2],
    ]
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    n_points = coords.shape[0]

    pred_chunks = []
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = coords[i : i + batch_size]
            cords_t = torch.from_numpy(batch_coords).to(device=device, dtype=torch.float32)
            out = model(cords_t, embedding)
            out = torch.sigmoid(out)
            pred_chunks.append(out.cpu().numpy())

    pred = np.concatenate(pred_chunks, axis=0)
    pred = pred.reshape(grid_shape[0], grid_shape[1], grid_shape[2], 4)
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)

    # voxelmap expects a 3D occupancy array (Z, X, Y)
    voxel_array = (pred[..., 0] > 0.5).astype(np.float32)
    vm = Model(array=voxel_array)
    vm.__dict__["_rgb"] = np.clip(pred[..., 1:], 0.0, 1.0).astype(np.float32)
    vm.objfile = "voxel_scene.obj"
    return vm


def render_voxel_slice(
    voxel_model: Model,
    axis: str = "z",
    pos: float | None = None,
    show: bool = True,
    title: str = "Voxel Slice",
) -> np.ndarray:
    """Render a 2D slice through the voxel model using `build()` + axis index."""
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Unknown axis '{axis}'")

    voxels = voxel_model.build()
    z, y, x = voxels.shape

    if axis == "x":
        idx = int(pos * x) if pos is not None else x // 2
        idx = max(0, min(x - 1, idx))
        img = voxels[:, :, idx].astype(np.float32)
    elif axis == "y":
        idx = int(pos * y) if pos is not None else y // 2
        idx = max(0, min(y - 1, idx))
        img = voxels[:, idx, :].astype(np.float32)
    else:
        idx = int(pos * z) if pos is not None else z // 2
        idx = max(0, min(z - 1, idx))
        img = voxels[idx, :, :].astype(np.float32)

    if show:
        import matplotlib.pyplot as plt

        plt.title(title)
        plt.axis("off")
        plt.imshow(img, cmap="gray", interpolation="none")
        plt.show()
    return img


def export_voxel_obj(
    voxel_model: Model,
    output_path: str,
) -> None:
    """Export voxel model as OBJ using voxelmap."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    voxel_model.save(output_path)
