import torch

from sdfmodel.models import SDFMLP, build_model, list_models
from sdfmodel.models.fourier_pe import FourierPositionEncoding


def test_model_registry() -> None:
    models = list_models()
    assert "sdf_mlp" in models

    model = build_model("sdf_mlp", in_features=3, hidden_features=64, num_layers=3)
    assert isinstance(model, SDFMLP)


def test_sdf_mlp_shape_contract() -> None:
    batch_size = 16
    in_dim = 3
    model = SDFMLP(
        in_features=in_dim, hidden_features=64, num_layers=3, use_fourier_pe=True
    )

    inputs = torch.randn(batch_size, in_dim)
    outputs = model(inputs)

    assert outputs.shape == (batch_size, 1)
    assert model.num_parameters > 0
    assert model.trainable_parameters == model.num_parameters


def test_sdf_mlp_gradient_flow() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    inputs = torch.randn(8, 3, requires_grad=True)
    outputs = model(inputs)

    loss = outputs.sum()
    loss.backward()

    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_fourier_pe_learnable() -> None:
    pe_static = FourierPositionEncoding(in_features=3, num_bands=6, learnable=False)
    assert "frequencies" not in dict(pe_static.named_parameters())

    pe_learnable = FourierPositionEncoding(in_features=3, num_bands=6, learnable=True)
    assert "frequencies" in dict(pe_learnable.named_parameters())

    inputs = torch.randn(4, 3, requires_grad=True)
    encoded = pe_learnable(inputs)
    loss = encoded.sum()
    loss.backward()

    assert pe_learnable.frequencies.grad is not None


def test_sdf_mlp_with_learnable_fourier() -> None:
    model = SDFMLP(
        in_features=3,
        hidden_features=32,
        num_layers=3,
        use_fourier_pe=True,
        fourier_learnable=True,
    )
    inputs = torch.randn(4, 3)
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    assert model.pe is not None
    assert model.pe.frequencies.grad is not None


def test_sdf_mlp_with_layernorm() -> None:
    model = SDFMLP(
        in_features=3,
        hidden_features=32,
        num_layers=3,
        norm_type="layernorm",
    )
    inputs = torch.randn(4, 3)
    outputs = model(inputs)
    assert outputs.shape == (4, 1)

    has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    assert has_layernorm


def test_sdf_mlp_with_weightnorm() -> None:
    model = SDFMLP(
        in_features=3,
        hidden_features=32,
        num_layers=3,
        norm_type="weightnorm",
    )
    inputs = torch.randn(4, 3)
    outputs = model(inputs)
    assert outputs.shape == (4, 1)
