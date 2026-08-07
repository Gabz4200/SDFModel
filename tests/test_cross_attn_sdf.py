import pytest
import torch

from sdfmodel.models import build_model, list_models
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


def test_cross_attn_sdf_registry() -> None:
    models = list_models()
    assert "cross_attn_sdf" in models

    model = build_model("cross_attn_sdf", hidden_dim=64, num_layers=2)
    assert isinstance(model, CrossAttnSDFModel)


def test_cross_attn_sdf_forward_shape_contract() -> None:
    batch_size = 4
    num_points = 100
    seq_len = 5
    hidden_dim = 64

    model = CrossAttnSDFModel(
        in_features=3,
        hidden_dim=hidden_dim,
        num_layers=6,
        num_heads=4,
        fourier_num_bands=6,
    )

    cords = torch.randn(batch_size, num_points, 3)
    embedding = torch.randn(batch_size, seq_len, hidden_dim)

    output = model(cords, embedding)

    assert output.shape == (batch_size, num_points, 1)
    assert model.num_parameters > 0
    assert model.trainable_parameters == model.num_parameters


def test_cross_attn_sdf_2d_cords_input() -> None:
    batch_size = 8
    seq_len = 3
    hidden_dim = 32

    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)

    cords = torch.randn(batch_size, 3)
    embedding = torch.randn(batch_size, seq_len, hidden_dim)

    output = model(cords, embedding)

    assert output.shape == (batch_size, 1)


def test_cross_attn_sdf_position_invariance() -> None:
    """Object token sequence has no positional encoding, so permuting tokens should yield invariant output."""
    torch.manual_seed(42)
    batch_size = 2
    num_points = 20
    seq_len = 6
    hidden_dim = 32

    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=3, num_heads=4).eval()

    cords = torch.randn(batch_size, num_points, 3)
    embedding = torch.randn(batch_size, seq_len, hidden_dim)

    perm = torch.randperm(seq_len)
    permuted_embedding = embedding[:, perm, :]

    with torch.no_grad():
        out_original = model(cords, embedding)
        out_permuted = model(cords, permuted_embedding)

    assert torch.allclose(out_original, out_permuted, atol=1e-5)


def test_cross_attn_sdf_gradient_flow() -> None:
    """Gradients must flow backward to cords, embedding, and all trainable parameters."""
    model = CrossAttnSDFModel(hidden_dim=32, num_layers=2, num_heads=2)

    cords = torch.randn(4, 50, 3, requires_grad=True)
    embedding = torch.randn(4, 5, 32, requires_grad=True)

    outputs = model(cords, embedding)
    loss = outputs.sum()
    loss.backward()

    assert cords.grad is not None
    assert cords.grad.shape == cords.shape
    assert embedding.grad is not None
    assert embedding.grad.shape == embedding.shape

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_cross_attn_sdf_batch_independence() -> None:
    """Output for batch item i should be independent of other items in the batch."""
    torch.manual_seed(123)
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2).eval()

    cords_0 = torch.randn(1, 15, 3)
    emb_0 = torch.randn(1, 4, hidden_dim)

    cords_1 = torch.randn(1, 15, 3)
    emb_1 = torch.randn(1, 4, hidden_dim)

    cords_batched = torch.cat([cords_0, cords_1], dim=0)
    emb_batched = torch.cat([emb_0, emb_1], dim=0)

    with torch.no_grad():
        out_single_0 = model(cords_0, emb_0)
        out_batched = model(cords_batched, emb_batched)

    assert torch.allclose(out_single_0, out_batched[0:1], atol=1e-5)


def test_cross_attn_sdf_learnable_embedding_optimization() -> None:
    """External learnable parameter created via factory method optimizes with SGD/Adam."""
    batch_size = 2
    seq_len = 4
    hidden_dim = 32

    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    learnable_emb = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=batch_size, seq_len=seq_len, hidden_dim=hidden_dim
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [learnable_emb], lr=0.01)

    cords = torch.randn(batch_size, 30, 3)
    targets = torch.zeros(batch_size, 30, 1)

    initial_emb_data = learnable_emb.clone()

    optimizer.zero_grad()
    predictions = model(cords, learnable_emb)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(learnable_emb, initial_emb_data)


def test_when_batch_sizes_mismatch_then_raises_value_error() -> None:
    model = CrossAttnSDFModel(hidden_dim=32, num_layers=2, num_heads=2)
    cords = torch.randn(4, 20, 3)
    embedding = torch.randn(2, 5, 32)

    with pytest.raises(ValueError, match="Batch size mismatch"):
        model(cords, embedding)


def test_when_coord_dim_mismatch_then_raises_value_error() -> None:
    model = CrossAttnSDFModel(in_features=3, hidden_dim=32, num_layers=2, num_heads=2)
    cords = torch.randn(4, 20, 4)  # 4 channels instead of 3
    embedding = torch.randn(4, 5, 32)

    with pytest.raises(ValueError, match="Expected coordinate feature dimension"):
        model(cords, embedding)


def test_when_embedding_hidden_dim_mismatch_then_raises_value_error() -> None:
    model = CrossAttnSDFModel(hidden_dim=32, num_layers=2, num_heads=2)
    cords = torch.randn(4, 20, 3)
    embedding = torch.randn(4, 5, 64)  # 64 instead of 32

    with pytest.raises(ValueError, match="Expected embedding hidden dimension"):
        model(cords, embedding)


def test_cross_attn_sdf_chunked_forward() -> None:
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(1, 4, hidden_dim)
    cords = torch.randn(1, 100, 3)

    out_unchunked = model(cords, embedding)
    out_chunked = model(cords, embedding, chunk_size=32)

    assert out_chunked.shape == out_unchunked.shape
    assert torch.allclose(out_unchunked, out_chunked, atol=1e-5)


def test_cross_attn_block_has_no_self_attention_layers() -> None:
    """Transformer block must not contain self-attention layers on object embeddings."""
    from sdfmodel.models.transformer_block import CrossAttentionTransformerBlock

    block = CrossAttentionTransformerBlock(hidden_dim=32, num_heads=2)
    assert not hasattr(block, "obj_self_attn")
    assert not hasattr(block, "obj_ffn")
    assert not hasattr(block, "obj_norm1")
    assert not hasattr(block, "obj_norm2")


def test_when_passing_multiple_cords_and_multiple_embeddings_2d_then_returns_per_cord_distances() -> (
    None
):
    """Passing 2D cords (N, 3) and 2D embeddings (S, D) outputs distance (N, 1)."""
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    cords = torch.randn(50, 3)
    embedding = torch.randn(4, hidden_dim)

    output = model(cords, embedding)

    assert output.shape == (50, 1)


def test_when_passing_multiple_cords_and_multiple_embeddings_3d_then_returns_per_cord_distances() -> (
    None
):
    """Passing 3D cords (B, N, 3) and 3D embeddings (B, S, D) outputs distance (B, N, 1)."""
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    cords = torch.randn(3, 50, 3)
    embedding = torch.randn(3, 4, hidden_dim)

    output = model(cords, embedding)

    assert output.shape == (3, 50, 1)


def test_when_passing_2d_cords_and_3d_embeddings_then_broadcasts_and_returns_batched_distances() -> (
    None
):
    """Passing 2D cords (N, 3) and 3D embeddings (B, S, D) broadcasts cords across batch."""
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    cords = torch.randn(50, 3)
    embedding = torch.randn(3, 4, hidden_dim)

    output = model(cords, embedding)

    assert output.shape == (3, 50, 1)


def test_when_passing_3d_cords_and_2d_embeddings_then_broadcasts_and_returns_batched_distances() -> (
    None
):
    """Passing 3D cords (B, N, 3) and 2D embeddings (S, D) broadcasts embeddings across batch."""
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    cords = torch.randn(3, 50, 3)
    embedding = torch.randn(4, hidden_dim)

    output = model(cords, embedding)

    assert output.shape == (3, 50, 1)
