import os

import numpy as np

from sdfmodel.models import SDFMLP, CrossAttnSDFModel
from sdfmodel.render import create_sdf3_wrapper, export_sdf_mesh, render_sdf_slice


def test_create_sdf3_wrapper_cross_attn() -> None:
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(1, 4, hidden_dim)

    sdf_obj = create_sdf3_wrapper(model, embedding=embedding)
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)

    values = sdf_obj(pts)
    assert values.shape == (2, 1) or values.shape == (2,)


def test_create_sdf3_wrapper_sdf_mlp() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    sdf_obj = create_sdf3_wrapper(model)

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    values = sdf_obj(pts)
    assert len(values) == 2


def test_render_sdf_slice_sampling() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    grid, extent, axes = render_sdf_slice(sdf_obj, resolution=32, z=0.0, show=False)
    assert grid.shape == (32, 32)
    assert axes == "YX"
    assert len(extent) == 4


def test_export_sdf_mesh(tmp_path) -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=2)
    sdf_obj = create_sdf3_wrapper(model)

    output_path = tmp_path / "test_sphere.stl"
    export_sdf_mesh(sdf_obj, str(output_path), step=0.5, verbose=False)
    assert os.path.exists(output_path)
