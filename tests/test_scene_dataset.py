import torch

from sdfmodel.datasets.scene_sdf import Scene4PrimitivesDataset, build_scene_dataloader


def test_scene_4_primitives_dataset_shape() -> None:
    dataset = Scene4PrimitivesDataset(num_samples=1024, points_per_item=128, seed=42)
    assert len(dataset) == 1024

    points, targets = dataset[0]
    assert points.shape == (128, 3)
    assert targets.shape == (128, 1)
    assert isinstance(points, torch.Tensor)
    assert isinstance(targets, torch.Tensor)


def test_scene_4_primitives_dataloader() -> None:
    loader = build_scene_dataloader(
        num_samples=64, points_per_item=256, batch_size=2, seed=42
    )

    batch_pts, batch_targets = next(iter(loader))
    assert batch_pts.shape == (2, 256, 3)
    assert batch_targets.shape == (2, 256, 1)
