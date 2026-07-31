import torch

from sdfmodel.models import SDFMLP, build_model, list_models


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
