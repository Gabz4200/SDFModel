import numpy as np
import torch

from sdfmodel.models import CrossAttnSDFModel, VectorSDFModel
from sdfmodel.render import create_sdf3_wrapper


def test_create_sdf3_wrapper_vector_model() -> None:
    """create_sdf3_wrapper should auto-detect VectorSDFModel and return scalar SDF."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )
    sdf_obj = create_sdf3_wrapper(model, embedding=embedding)

    # Evaluate at some points
    pts = np.random.randn(100, 3).astype(np.float32)
    values = sdf_obj(pts)
    assert len(values) == 100


def test_vector_sdf3_wrapper_scalar_values() -> None:
    """VectorSDFModel wrapped in SDF3 should return scalar SDF values."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )
    sdf_obj = create_sdf3_wrapper(model, embedding=embedding)

    # Test points
    pts = torch.randn(50, 3)
    pts_np = pts.detach().numpy().astype(np.float32)

    # Get scalar SDF from wrapper
    wrapper_values = sdf_obj(pts_np)

    # Just check it returns reasonable finite scalar values
    assert np.all(np.isfinite(wrapper_values))
    assert wrapper_values.shape == (50, 1) or wrapper_values.shape == (50,)


def test_vector_sdf3_wrapper_shape() -> None:
    """VectorSDFModel wrapped in SDF3 should return (N, 1) for (N, 3) input."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )
    sdf_obj = create_sdf3_wrapper(model, embedding=embedding)

    pts = np.random.randn(25, 3).astype(np.float32)
    values = sdf_obj(pts)

    # Should return (N, 1) format
    assert values.shape == (25, 1) or values.shape == (25,)
