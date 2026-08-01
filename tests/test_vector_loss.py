import torch

from sdfmodel.engine.metrics import compute_vector_sdf_loss
from sdfmodel.models import VectorSDFModel
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


def test_compute_vector_sdf_loss_basic() -> None:
    """Vector loss should return dict with expected keys and positive loss."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    points = torch.randn(10, 3)
    target_sdf = torch.randn(10, 1)
    target_normals = torch.randn(10, 3)
    # Normalize target normals
    target_normals = target_normals / target_normals.norm(dim=-1, keepdim=True)

    loss_dict = compute_vector_sdf_loss(
        model=model,
        points=points,
        target_sdf=target_sdf,
        target_normals=target_normals,
        embedding=embedding,
    )

    assert "loss" in loss_dict
    assert "vector_l2_loss" in loss_dict
    assert "cosine_loss" in loss_dict
    assert "magnitude_mse_loss" in loss_dict
    assert "eikonal_loss" in loss_dict
    assert "normal_loss" in loss_dict

    assert loss_dict["loss"].ndim == 0
    assert loss_dict["loss"].item() >= 0.0


def test_compute_vector_sdf_loss_gradient_flow() -> None:
    """Vector loss gradients should flow to model parameters."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    points = torch.randn(10, 3)
    target_sdf = torch.randn(10, 1)
    target_normals = torch.randn(10, 3)
    target_normals = target_normals / target_normals.norm(dim=-1, keepdim=True)

    loss_dict = compute_vector_sdf_loss(
        model=model,
        points=points,
        target_sdf=target_sdf,
        target_normals=target_normals,
        embedding=embedding,
    )

    loss = loss_dict["loss"]
    loss.backward()

    # Check at least one parameter has gradient
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad


def test_compute_vector_sdf_loss_without_target_normals() -> None:
    """Vector loss should work when target_normals is None."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    points = torch.randn(10, 3)
    target_sdf = torch.randn(10, 1)

    loss_dict = compute_vector_sdf_loss(
        model=model,
        points=points,
        target_sdf=target_sdf,
        target_normals=None,
        embedding=embedding,
    )

    assert "loss" in loss_dict
    assert loss_dict["loss"].ndim == 0
    assert loss_dict["loss"].item() >= 0.0
