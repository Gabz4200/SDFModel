from typing import Any, cast

import numpy as np
import sdf
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

    batch = next(iter(loader))
    assert len(batch) in (2, 3)
    batch_pts, batch_targets = batch[0], batch[1]
    assert batch_pts.shape == (2, 256, 3)
    assert batch_targets.shape == (2, 256, 1)
    if len(batch) == 3:
        batch_normals = batch[2]
        assert batch_normals.shape == (2, 256, 3)


def test_scene_4_primitives_dataset_normals() -> None:
    dataset = Scene4PrimitivesDataset(
        num_samples=32, points_per_item=64, seed=42, return_normals=True
    )
    pts, targets, normals = dataset[0]
    assert pts.shape == (64, 3)
    assert targets.shape == (64, 1)
    assert normals.shape == (64, 3)

    norms = normals.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_scene_4_primitives_surface_band_sampling() -> None:
    """Near-surface points must lie within surface_eps of the zero level set.

    Unlike Gaussian-around-centers sampling, the density-based rejection
    sampler should put most near points on the analytic surface band —
    including the torus hole rim.
    """
    eps = 0.1
    dataset = Scene4PrimitivesDataset(
        num_samples=4,
        points_per_item=64,
        seed=7,
        return_normals=True,
        surface_eps=eps,
    )
    assert dataset.surface_eps == eps

    n_near = dataset.points_per_item // 2
    near_points = dataset.points_data[:, :n_near, :].reshape(-1, 3)
    f = dataset.scene(near_points).squeeze(-1)

    frac_in_band = float(np.mean(np.abs(f) < eps))
    assert frac_in_band > 0.8, f"Only {frac_in_band:.2f} of near points in surface band"


def test_scene_4_primitives_dataloader_surface_eps() -> None:
    """build_scene_dataloader must forward surface_eps to the dataset."""
    loader = build_scene_dataloader(
        num_samples=1,
        points_per_item=16,
        batch_size=2,
        return_normals=True,
        surface_eps=0.05,
    )
    assert loader.dataset.surface_eps == 0.05  # type: ignore[attr-defined]


def test_scene_4_primitives_chaos_game_sampling() -> None:
    """Chaos-game near points must concentrate in the surface band and explore.

    The chaos game projects particles onto the zero level set (so nearly all
    land inside the band) while jitter keeps them spread over the surface
    instead of collapsing to a single point.
    """
    eps = 0.1
    dataset = Scene4PrimitivesDataset(
        num_samples=4,
        points_per_item=64,
        seed=7,
        return_normals=True,
        surface_eps=eps,
        sampler="chaos_game",
    )
    assert dataset.sampler == "chaos_game"

    n_near = dataset.points_per_item // 2
    near_points = dataset.points_data[:, :n_near, :].reshape(-1, 3)
    f = dataset.scene(near_points).squeeze(-1)

    frac_in_band = float(np.mean(np.abs(f) < eps))
    assert frac_in_band > 0.95, f"Only {frac_in_band:.2f} of chaos points in surface band"

    # Exploration: points must not collapse onto a single surface location
    spread = float(np.std(near_points, axis=0).mean())
    assert spread > 0.05, f"Chaos points collapsed (spread {spread:.3f})"


def test_scene_4_primitives_chaos_game_covers_torus_rim() -> None:
    """Chaos-game sampling must reach the torus hole rim interior.

    This is the donut-hole regression: the rim interior wall is a thin feature
    that uniform rejection sampling statistically misses, starving the model of
    samples on the hole boundary.
    """
    dataset = Scene4PrimitivesDataset(
        num_samples=1,
        points_per_item=4096,
        seed=7,
        sampler="chaos_game",
        return_normals=True,
    )
    n_near = dataset.points_per_item // 2
    near_points = dataset.points_data[0, :n_near, :]

    # Rebuild the torus component (center (-0.4, 0.4, 0), ring r=0.25, tube R=0.08)
    sdf_api = cast(Any, sdf)
    torus = sdf_api.torus(0.25, 0.08).translate((-0.4, 0.4, 0.0))
    f_torus = torus(near_points).squeeze(-1)
    on_torus = np.abs(f_torus) < 0.01

    # Inner wall of the tube (the side facing the hole): distance to the torus
    # center axis < 0.24 (centerline is 0.25, hole edge at 0.17).
    dist_axis = np.sqrt((near_points[:, 0] + 0.4) ** 2 + (near_points[:, 1] - 0.4) ** 2)
    rim_hits = int(np.sum(on_torus & (dist_axis < 0.24)))

    assert rim_hits > 10, f"Chaos game found only {rim_hits} points on the torus rim interior"


def test_scene_dataloader_sampler_param() -> None:
    """build_scene_dataloader must forward sampler and chaos_iters."""
    loader = build_scene_dataloader(
        num_samples=1,
        points_per_item=16,
        batch_size=2,
        sampler="rejection",
        chaos_iters=2,
    )
    assert loader.dataset.sampler == "rejection"  # type: ignore[attr-defined]
    assert loader.dataset.chaos_iters == 2  # type: ignore[attr-defined]

    import pytest

    with pytest.raises(ValueError):
        Scene4PrimitivesDataset(num_samples=1, points_per_item=16, sampler="bogus")

