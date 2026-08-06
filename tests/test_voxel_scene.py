from sdfmodel.datasets.voxel_scene import VoxelSceneDataset


def test_voxel_dataset_loads_donut_file() -> None:
    dataset = VoxelSceneDataset(
        voxel_path="data/donutvoxel.vox",
        num_samples=4,
        points_per_item=32,
        seed=42,
    )

    assert len(dataset) == 4

    pts, targets = dataset[0]
    assert pts.shape == (32, 3)
    assert targets.shape == (32, 4)

    assert (targets[:, 0] >= 0.0).all()
    assert (targets[:, 0] <= 1.0).all()
    assert (targets[:, 1:] >= 0.0).all()
    assert (targets[:, 1:] <= 1.0).all()


def test_voxel_dataset_has_occupied_and_empty() -> None:
    dataset = VoxelSceneDataset(
        voxel_path="data/donutvoxel.vox",
        num_samples=8,
        points_per_item=64,
        seed=0,
    )

    _, targets = dataset[0]
    exist = targets[:, 0]
    assert exist.min() <= 0.0
    assert exist.max() >= 1.0
