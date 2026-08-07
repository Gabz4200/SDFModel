import torch

from sdfmodel.engine.metrics import compute_vector_sdf_loss
from sdfmodel.models import VectorSDFModel
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel
from sdfmodel.models.fourier_pe import FourierPositionEncoding


def test_vector_sdf_direction_head_gets_gradient() -> None:
    """The learned direction head must receive gradients from cosine loss alone.

    Regression test: the old autograd-based vector output (create_graph=False)
    had no path back to the network for direction supervision.
    """
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    cords = torch.randn(10, 3)
    output = model(cords, embedding)
    # Direction-only supervision: align vector direction to a fixed target
    target_dir = torch.randn(10, 3)
    target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
    pred_dir = output / output.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    loss = torch.mean(
        1.0 - torch.nn.functional.cosine_similarity(pred_dir, target_dir, dim=-1)
    )

    loss.backward()

    assert model.vector_head.weight.grad is not None
    assert torch.abs(model.vector_head.weight.grad).sum() > 0.0
    # Direction supervision must reach the shared backbone, not just the head
    backbone_layer = model.cords_mlp[0]
    assert isinstance(backbone_layer, torch.nn.Linear)
    assert backbone_layer.weight.grad is not None
    assert torch.abs(backbone_layer.weight.grad).sum() > 0.0


def test_vector_sdf_vector_magnitude_equals_scalar() -> None:
    """|v| must equal |f| by construction (vector = normalized_dir * dist)."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    cords = torch.randn(10, 3)
    vector = model(cords, embedding)
    scalar = model.predict_scalar(cords, embedding)

    assert torch.allclose(vector.norm(dim=-1), scalar.abs().squeeze(-1), atol=1e-5)


def test_vector_sdf_scene_token_position_invariance() -> None:
    """With a scene summary token, permuting object tokens must stay invariant."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2, use_scene_token=True)
    assert hasattr(model, "scene_token")
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    cords = torch.randn(5, 3)
    out_original = model(cords, embedding)
    perm = torch.tensor([2, 0, 3, 1])
    out_permuted = model(cords, embedding[:, perm, :])
    assert torch.allclose(out_original, out_permuted, atol=1e-5)


def test_vector_sdf_scene_token_grad_flow() -> None:
    """Scene token parameter must receive gradients."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2, use_scene_token=True)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )
    cords = torch.randn(10, 3)
    output = model(cords, embedding)
    output.sum().backward()
    assert model.scene_token.grad is not None
    assert torch.abs(model.scene_token.grad).sum() > 0.0


def test_fourier_pe_active_bands_masking() -> None:
    """active_bands must zero out high-frequency features."""
    pe = FourierPositionEncoding(in_features=3, num_bands=8)
    x = torch.randn(4, 3)

    pe.active_bands = None
    full = pe(x)
    pe.active_bands = 2
    masked = pe(x)

    # Raw coordinates (first 3 dims) always present
    assert torch.allclose(full[..., :3], masked[..., :3])

    # Layout: [x(3), sin(C*B), cos(C*B)] with per-coordinate band blocks.
    c, b = 3, pe.num_bands
    sin_block = full[..., 3 : 3 + c * b]
    cos_block = full[..., 3 + c * b : 3 + 2 * c * b]
    msin_block = masked[..., 3 : 3 + c * b]
    mcos_block = masked[..., 3 + c * b : 3 + 2 * c * b]

    for coord in range(c):
        base = coord * b
        # Kept bands (0, 1) equal full output
        assert torch.allclose(
            msin_block[..., base : base + 2], sin_block[..., base : base + 2]
        )
        assert torch.allclose(
            mcos_block[..., base : base + 2], cos_block[..., base : base + 2]
        )
        # Zeroed bands (2..7) are exactly zero
        assert torch.all(msin_block[..., base + 2 : base + b] == 0.0)
        assert torch.all(mcos_block[..., base + 2 : base + b] == 0.0)


def test_fourier_pe_active_bands_reaches_model() -> None:
    """SceneTrainer-style annealing: setting active_bands on the model's PE changes outputs."""
    model = CrossAttnSDFModel(hidden_dim=32, num_layers=2, fourier_num_bands=8)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )
    cords = torch.randn(2, 10, 3)

    model.fourier_pe.active_bands = 4
    out_mid = model(cords, embedding)
    model.fourier_pe.active_bands = 8
    out_full = model(cords, embedding)
    model.fourier_pe.active_bands = 0
    out_none = model(cords, embedding)

    assert not torch.allclose(out_mid, out_full, atol=1e-6)
    assert not torch.allclose(out_none, out_full, atol=1e-6)
    assert out_none.shape == out_full.shape == (2, 10, 1)


def test_compute_vector_sdf_loss_consistency_key() -> None:
    """compute_vector_sdf_loss must return a consistency_loss term."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )
    points = torch.randn(10, 3)
    target_sdf = torch.randn(10, 1) * 0.5
    target_normals = torch.randn(10, 3)
    target_normals = target_normals / target_normals.norm(dim=-1, keepdim=True)

    loss_dict = compute_vector_sdf_loss(
        model=model,
        points=points,
        target_sdf=target_sdf,
        target_normals=target_normals,
        embedding=embedding,
        w_consistency=0.3,
    )

    assert "consistency_loss" in loss_dict
    assert loss_dict["consistency_loss"].ndim == 0
    total = loss_dict["loss"]
    total.backward()
    assert model.vector_head.weight.grad is not None
