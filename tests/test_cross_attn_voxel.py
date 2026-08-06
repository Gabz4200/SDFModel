import torch

from sdfmodel.models import build_model, list_models
from sdfmodel.models.cross_attn_voxel import CrossAttnVoxelModel


def test_voxel_model_registry() -> None:
    models = list_models()
    assert "cross_attn_voxel" in models

    model = build_model("cross_attn_voxel", hidden_dim=64, num_layers=2)
    assert isinstance(model, CrossAttnVoxelModel)


def test_voxel_model_forward_shape_contract() -> None:
    hidden_dim = 32
    model = CrossAttnVoxelModel(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        fourier_num_bands=4,
    )

    cords = torch.randn(8, 3)
    embedding = torch.randn(1, 4, hidden_dim)
    output = model(cords, embedding)

    assert output.shape == (1, 8, 4)
    assert output.dtype == torch.float32


def test_voxel_model_output_in_unit_range() -> None:
    hidden_dim = 32
    model = CrossAttnVoxelModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)

    cords = torch.randn(16, 3)
    embedding = torch.randn(1, 8, hidden_dim)
    output = model(cords, embedding)

    assert (output >= 0.0).all()
    assert (output <= 1.0).all()


def test_voxel_model_4d_output_per_coordinate() -> None:
    hidden_dim = 32
    model = CrossAttnVoxelModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)

    cords = torch.randn(10, 3)
    embedding = torch.randn(1, 4, hidden_dim)
    output = model(cords, embedding)

    assert output.shape == (1, 10, 4)


def test_voxel_model_gradient_flow() -> None:
    hidden_dim = 32
    model = CrossAttnVoxelModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)

    cords = torch.randn(8, 3, requires_grad=True)
    embedding = torch.randn(1, 4, hidden_dim, requires_grad=True)
    output = model(cords, embedding)
    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
    assert cords.grad is not None
    assert embedding.grad is not None


def test_voxel_model_batch_independence() -> None:
    hidden_dim = 32
    model = CrossAttnVoxelModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)

    embedding = torch.randn(1, 4, hidden_dim)

    cords_0 = torch.randn(1, 3)
    cords_1 = torch.randn(1, 3)
    cords_batched = torch.stack([cords_0, cords_1])

    with torch.no_grad():
        out_0 = model(cords_0, embedding)
        out_1 = model(cords_1, embedding)
        out_batched = model(cords_batched, embedding)

    assert torch.allclose(out_batched[0], out_0, atol=1e-5)
    assert torch.allclose(out_batched[1], out_1, atol=1e-5)


def test_voxel_model_chunked_forward() -> None:
    hidden_dim = 32
    model = CrossAttnVoxelModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    model.eval()

    cords = torch.randn(1, 64, 3)
    embedding = torch.randn(1, 4, hidden_dim)

    with torch.no_grad():
        out_unchunked = model(cords, embedding)
        out_chunked = model(cords, embedding, chunk_size=16)

    assert torch.allclose(out_unchunked, out_chunked, atol=1e-5)
