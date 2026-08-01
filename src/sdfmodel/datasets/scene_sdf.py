from typing import Any, cast

import numpy as np
import sdf
import torch
from torch.utils.data import DataLoader, Dataset


def create_4_primitives_scene() -> sdf.d3.SDF3:
    """Create a 3D scene containing 4 primitives: sphere, box, torus, capped_cylinder."""
    sdf_api = cast(Any, sdf)
    s1 = sdf_api.sphere(0.35).translate((-0.4, -0.4, 0.0))
    s2 = sdf_api.box((0.3, 0.3, 0.3)).translate((0.4, -0.4, 0.0))
    s3 = sdf_api.torus(0.25, 0.08).translate((-0.4, 0.4, 0.0))
    s4 = sdf_api.capped_cylinder(-sdf.Z * 0.25, sdf.Z * 0.25, 0.2).translate(
        (0.4, 0.4, 0.0)
    )

    return s1 | s2 | s3 | s4


class Scene4PrimitivesDataset(Dataset):
    """Dataset sampling near-surface and uniform 3D points for a 4-primitive scene."""

    def __init__(
        self,
        num_samples: int = 1024,
        points_per_item: int = 256,
        bounds: float = 1.0,
        seed: int = 42,
        return_normals: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.points_per_item = points_per_item
        self.return_normals = return_normals
        self.scene = create_4_primitives_scene()

        rng = np.random.default_rng(seed)

        primitive_centers = np.array(
            [
                [-0.4, -0.4, 0.0],
                [0.4, -0.4, 0.0],
                [-0.4, 0.4, 0.0],
                [0.4, 0.4, 0.0],
            ],
            dtype=np.float32,
        )

        n_near = points_per_item // 2
        n_uniform = points_per_item - n_near

        sampled_points = []
        for _ in range(num_samples):
            center_indices = rng.choice(4, size=n_near)
            near = primitive_centers[center_indices] + rng.normal(
                0.0, 0.25, size=(n_near, 3)
            ).astype(np.float32)
            uniform = rng.uniform(-bounds, bounds, size=(n_uniform, 3)).astype(
                np.float32
            )

            pts = np.vstack([near, uniform])
            sampled_points.append(pts)

        self.points_data = np.array(sampled_points, dtype=np.float32)

        flat_points = self.points_data.reshape(-1, 3)
        flat_sdfs = self.scene(flat_points).squeeze(-1).astype(np.float32)
        self.sdf_data = flat_sdfs.reshape(num_samples, points_per_item, 1)

        h = 1e-4
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        dx = (self.scene(flat_points + h * ex) - self.scene(flat_points - h * ex)) / (
            2 * h
        )
        dy = (self.scene(flat_points + h * ey) - self.scene(flat_points - h * ey)) / (
            2 * h
        )
        dz = (self.scene(flat_points + h * ez) - self.scene(flat_points - h * ez)) / (
            2 * h
        )

        flat_grads = np.hstack([dx, dy, dz]).astype(np.float32)
        norms = np.linalg.norm(flat_grads, axis=-1, keepdims=True)
        flat_normals = flat_grads / np.maximum(norms, 1e-8)
        self.normal_data = flat_normals.reshape(num_samples, points_per_item, 3)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pts = torch.from_numpy(self.points_data[index])
        targets = torch.from_numpy(self.sdf_data[index])
        if self.return_normals:
            normals = torch.from_numpy(self.normal_data[index])
            return pts, targets, normals
        return pts, targets


def build_scene_dataloader(
    num_samples: int = 512,
    points_per_item: int = 256,
    batch_size: int = 2,
    seed: int = 42,
    return_normals: bool = False,
) -> DataLoader:
    dataset = Scene4PrimitivesDataset(
        num_samples=num_samples,
        points_per_item=points_per_item,
        seed=seed,
        return_normals=return_normals,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
