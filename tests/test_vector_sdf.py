import torch

from sdfmodel.models import VectorSDFModel, build_model, list_models
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


def test_vector_sdf_model_registration() -> None:
    """VectorSDFModel should be registered as 'vector_sdf'."""
    models = list_models()
    assert "vector_sdf" in models
    model = build_model("vector_sdf", hidden_dim=32, num_layers=2)
    assert isinstance(model, VectorSDFModel)


def test_vector_sdf_model_is_subclass_of_cross_attn_sdf() -> None:
    """VectorSDFModel should inherit from CrossAttnSDFModel."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    assert isinstance(model, CrossAttnSDFModel)
    assert isinstance(model, VectorSDFModel)


def test_vector_sdf_forward_shape_contract() -> None:
    """VectorSDFModel.forward should return (N, 3) or (B, N, 3) vectors."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    # 2D input: (N, 3) -> (N, 3)
    cords_2d = torch.randn(10, 3)
    output_2d = model(cords_2d, embedding)
    assert output_2d.shape == (10, 3)

    # 3D input: (B, N, 3) -> (B, N, 3)
    cords_3d = torch.randn(2, 10, 3)
    output_3d = model(cords_3d, embedding)
    assert output_3d.shape == (2, 10, 3)


def test_vector_sdf_predict_scalar_shape() -> None:
    """VectorSDFModel.predict_scalar should return scalar SDF values."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    # 2D input: (N, 3) -> (N, 1)
    cords_2d = torch.randn(10, 3)
    output_2d = model.predict_scalar(cords_2d, embedding)
    assert output_2d.shape == (10, 1)

    # 3D input: (B, N, 3) -> (B, N, 1)
    cords_3d = torch.randn(2, 10, 3)
    output_3d = model.predict_scalar(cords_3d, embedding)
    assert output_3d.shape == (2, 10, 1)


def test_vector_sdf_gradient_flow() -> None:
    """Gradients must flow backward through VectorSDFModel."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    cords = torch.randn(10, 3, requires_grad=True)
    output = model(cords, embedding)
    loss = output.sum()
    loss.backward()

    # Check model parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

    # Check embedding has gradient
    assert embedding.grad is not None

    # Check input has gradient
    assert cords.grad is not None


def test_vector_sdf_chunked_forward() -> None:
    """VectorSDFModel should support chunked forward passes."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    # Large batch
    cords = torch.randn(100, 3)

    # Without chunking
    out_unchunked = model(cords, embedding, chunk_size=None)

    # With chunking
    out_chunked = model(cords, embedding, chunk_size=25)

    assert torch.allclose(out_unchunked, out_chunked, atol=1e-5)


def test_vector_sdf_2d_cords_input() -> None:
    """VectorSDFModel should handle 2D coordinate input (N, 3)."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    batch_size = 8
    cords = torch.randn(batch_size, 3)
    output = model(cords, embedding)
    assert output.shape == (batch_size, 3)


def test_vector_sdf_batch_independence() -> None:
    """Output for batch item i should be independent of other items in the batch."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    # Single item
    single_cords = torch.randn(1, 3)
    out_single = model(single_cords, embedding)

    # Batched
    batch_cords = torch.randn(3, 3)
    batch_cords[0] = single_cords[0]
    out_batched = model(batch_cords, embedding)

    assert torch.allclose(out_single[0:1], out_batched[0:1], atol=1e-5)


def test_vector_sdf_position_invariance() -> None:
    """Object token sequence has no positional encoding, so permuting tokens should yield invariant output."""
    model = VectorSDFModel(hidden_dim=32, num_layers=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=32
    )

    cords = torch.randn(5, 3)
    out_original = model(cords, embedding)

    # Permute embedding tokens
    perm = torch.tensor([2, 0, 3, 1])
    embedding_permuted = embedding[:, perm, :]
    out_permuted = model(cords, embedding_permuted)

    assert torch.allclose(out_original, out_permuted, atol=1e-5)
