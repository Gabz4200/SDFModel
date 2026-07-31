import numpy as np
import sdf
import torch
from torch.utils.data import DataLoader, Dataset


def create_4_primitives_scene() -> sdf.d3.SDF3:
    """Create a 3D scene containing 4 primitives: sphere, box, torus, capped_cylinder."""
    s1 = sdf.sphere(0.35).translate((-0.4, -0.4, 0.0))
    s2 = sdf.box((0.3, 0.3, 0.3)).translate((0.4, -0.4, 0.0))
    s3 = sdf.torus(0.25, 0.08).translate((-0.4, 0.4, 0.0))
    s4 = sdf.capped_cylinder(-sdf.Z * 0.25, sdf.Z * 0.25, 0.2).translate(
        (0.4, 0.4, 0.0)
    )

    return s1 | s2 | s3 | s4


class Scene4PrimitivesDataset(Dataset):
    """Dataset sampling points and ground-truth SDF distances for a 4-primitive scene."""

    def __init__(
        self,
        num_samples: int = 1024,
        points_per_item: int = 256,
        bounds: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.points_per_item = points_per_item
        self.scene = create_4_primitives_scene()

        rng = np.random.default_rng(seed)
        self.points_data = rng.uniform(
            -bounds, bounds, size=(num_samples, points_per_item, 3)
        ).astype(np.float32)

        # Precompute SDF values for each sample batch
        flat_points = self.points_data.reshape(-1, 3)
        flat_sdfs = self.scene(flat_points).squeeze(-1).astype(np.float32)
        self.sdf_data = flat_sdfs.reshape(num_samples, points_per_item, 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pts = torch.from_numpy(self.points_data[index])
        targets = torch.from_numpy(self.sdf_data[index])
        return pts, targets


def build_scene_dataloader(
    num_samples: int = 512,
    points_per_item: int = 256,
    batch_size: int = 2,
    seed: int = 42,
) -> DataLoader:
    dataset = Scene4PrimitivesDataset(
        num_samples=num_samples,
        points_per_item=points_per_item,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
