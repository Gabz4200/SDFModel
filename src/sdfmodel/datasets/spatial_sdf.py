import torch
from torch.utils.data import DataLoader, Dataset

from sdfmodel.utils.config import DatasetConfig


class SyntheticSDFDataset(Dataset):
    """Synthetic PyTorch Dataset for 3D Sphere Signed Distance Function.

    Analytical SDF formula for sphere centered at origin with radius R:
    SDF(p) = ||p||_2 - R
    """

    def __init__(
        self, num_samples: int = 4096, radius: float = 1.0, seed: int = 42
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.radius = radius

        generator = torch.Generator().manual_seed(seed)
        self.points = (
            (torch.rand(num_samples, 3, generator=generator) - 0.5) * 3.0 * radius
        )
        norms = torch.norm(self.points, dim=-1, keepdim=True)
        self.sdfs = norms - radius

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points[index], self.sdfs[index]


def build_dataloaders(
    config: DatasetConfig, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    num_train = int(config.num_samples * 0.8)
    num_val = config.num_samples - num_train

    train_ds = SyntheticSDFDataset(
        num_samples=num_train, radius=config.radius, seed=seed
    )
    val_ds = SyntheticSDFDataset(
        num_samples=num_val, radius=config.radius, seed=seed + 1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
