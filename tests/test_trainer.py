import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from sdfmodel.datasets import build_dataloaders
from sdfmodel.engine import Trainer, compute_eikonal_loss
from sdfmodel.models import SDFMLP
from sdfmodel.utils.config import ExperimentConfig


def test_trainer_fit_and_checkpointing() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = ExperimentConfig()
        config.dataset.num_samples = 128
        config.dataset.batch_size = 32
        config.training.epochs = 2
        config.training.checkpoint_dir = tmp_dir
        config.training.device = "cpu"

        model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
        train_loader, val_loader = build_dataloaders(config.dataset, seed=42)

        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader, config=config
        )
        metrics = trainer.fit()

        assert "mse" in metrics
        assert "psnr" in metrics
        assert Path(tmp_dir, "best.pt").exists()

        # Test loading checkpoint
        new_model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
        trainer_load = Trainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        trainer_load.load_checkpoint(Path(tmp_dir, "best.pt"))

        p = torch.randn(4, 3)
        assert torch.allclose(model(p), new_model(p), atol=1e-5)


def test_eikonal_loss_computation() -> None:
    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    points = torch.randn(10, 3)

    # Test default directional finite difference with adaptive epsilon
    loss_fd = compute_eikonal_loss(model, points, use_autograd=False)
    assert loss_fd.ndim == 0
    assert loss_fd.item() >= 0.0

    # Test gradient backward flow for finite difference
    loss_fd.backward()
    param = next(model.parameters())
    assert param.grad is not None

    # Test autograd mode
    model.zero_grad()
    loss_ag = compute_eikonal_loss(model, points, use_autograd=True)
    assert loss_ag.ndim == 0
    assert loss_ag.item() >= 0.0

    loss_ag.backward()
    assert param.grad is not None


def test_eikonal_loss_with_embedding() -> None:
    from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel

    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embedding = CrossAttnSDFModel.create_learnable_embedding(1, 4, hidden_dim)
    points = torch.randn(2, 10, 3)

    loss_fd = compute_eikonal_loss(
        model, points, embedding=embedding, use_autograd=False
    )
    assert loss_fd.ndim == 0
    assert loss_fd.item() >= 0.0

    loss_ag = compute_eikonal_loss(
        model, points, embedding=embedding, use_autograd=True
    )
    assert loss_ag.ndim == 0
    assert loss_ag.item() >= 0.0


def test_compute_sdf_normals_methods() -> None:
    from sdfmodel.engine.metrics import compute_sdf_normals

    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    points = torch.randn(10, 3)

    for method in ("central", "tetrahedron", "autograd"):
        normals = compute_sdf_normals(model, points, method=method)
        assert normals.shape == (10, 3)
        norms = normals.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_compute_normal_loss() -> None:
    from sdfmodel.engine.metrics import compute_normal_loss

    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    points = torch.randn(10, 3)
    target_normals = F.normalize(torch.randn(10, 3), dim=-1)

    loss = compute_normal_loss(model, points, target_normals, method="central")
    assert loss.ndim == 0
    assert loss.item() >= 0.0

    loss.backward()
    param = next(model.parameters())
    assert param.grad is not None


def test_compute_combined_sdf_loss() -> None:
    from sdfmodel.engine.metrics import compute_combined_sdf_loss

    model = SDFMLP(in_features=3, hidden_features=32, num_layers=3)
    points = torch.randn(10, 3)
    target_sdf = torch.randn(10, 1)
    target_normals = F.normalize(torch.randn(10, 3), dim=-1)

    loss_dict = compute_combined_sdf_loss(
        model=model,
        points=points,
        target_sdf=target_sdf,
        target_normals=target_normals,
        w_distance=1.0,
        w_l1=0.5,
        w_eikonal=0.1,
        w_normal=0.2,
    )

    assert "loss" in loss_dict
    assert "mse_loss" in loss_dict
    assert "l1_loss" in loss_dict
    assert "eikonal_loss" in loss_dict
    assert "normal_loss" in loss_dict

    total_loss = loss_dict["loss"]
    assert total_loss.ndim == 0
    assert total_loss.item() >= 0.0

    expected_val = (
        1.0 * loss_dict["mse_loss"].item()
        + 0.5 * loss_dict["l1_loss"].item()
        + 0.1 * loss_dict["eikonal_loss"].item()
        + 0.2 * loss_dict["normal_loss"].item()
    )
    assert abs(total_loss.item() - expected_val) < 1e-4

    total_loss.backward()
    param = next(model.parameters())
    assert param.grad is not None

