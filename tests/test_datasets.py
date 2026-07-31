import torch

from sdfmodel.datasets import SyntheticSDFDataset, build_dataloaders
from sdfmodel.utils.config import DatasetConfig


def test_synthetic_sdf_dataset_shapes() -> None:
    num_samples = 100
    ds = SyntheticSDFDataset(num_samples=num_samples, radius=1.0, seed=42)

    assert len(ds) == num_samples
    point, sdf = ds[0]
    assert point.shape == (3,)
    assert sdf.shape == (1,)
    assert isinstance(point, torch.Tensor)
    assert isinstance(sdf, torch.Tensor)


def test_build_dataloaders() -> None:
    config = DatasetConfig(num_samples=200, batch_size=32, num_workers=0)
    train_loader, val_loader = build_dataloaders(config, seed=42)

    train_batch = next(iter(train_loader))
    points, sdfs = train_batch
    assert points.shape == (32, 3)
    assert sdfs.shape == (32, 1)

    val_batch = next(iter(val_loader))
    v_points, _ = val_batch
    assert v_points.shape[0] <= 40
