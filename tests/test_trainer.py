import tempfile
from pathlib import Path

import torch

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
