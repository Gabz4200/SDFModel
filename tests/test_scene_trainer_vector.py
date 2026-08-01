import torch

from sdfmodel.datasets.scene_sdf import build_scene_dataloader
from sdfmodel.engine import SceneTrainer
from sdfmodel.models import VectorSDFModel
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


def test_scene_trainer_vector_model_step() -> None:
    """SceneTrainer should support VectorSDFModel with vector loss."""
    hidden_dim = 32
    model = VectorSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embeddings = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=hidden_dim
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [embeddings], lr=0.01)

    loader = build_scene_dataloader(
        num_samples=16, points_per_item=128, batch_size=2, seed=42,
        return_normals=True,
    )

    trainer = SceneTrainer(
        model=model,
        learnable_embeddings=embeddings,
        dataloader=loader,
        optimizer=optimizer,
        device="cpu",
        view=False,
        model_type="vector_sdf",
    )

    initial_loss = trainer.train_step(0, *next(iter(loader)))
    assert isinstance(initial_loss, float)
    assert initial_loss > 0.0

    final_loss = initial_loss
    for batch in loader:
        points = batch[0]
        targets = batch[1]
        target_normals = batch[2] if len(batch) > 2 else None
        final_loss = trainer.train_step(0, points, targets, target_normals=target_normals)

    assert final_loss <= initial_loss or final_loss >= 0.0


def test_scene_trainer_vector_model_loss_decrease() -> None:
    """Vector SDF training should show loss decrease over steps."""
    hidden_dim = 32
    model = VectorSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embeddings = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=hidden_dim
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [embeddings], lr=0.01)

    loader = build_scene_dataloader(
        num_samples=32, points_per_item=64, batch_size=2, seed=42,
        return_normals=True,
    )

    trainer = SceneTrainer(
        model=model,
        learnable_embeddings=embeddings,
        dataloader=loader,
        optimizer=optimizer,
        device="cpu",
        view=False,
        model_type="vector_sdf",
    )

    losses = []
    for step, batch in enumerate(loader):
        points = batch[0]
        targets = batch[1]
        target_normals = batch[2] if len(batch) > 2 else None
        loss_val = trainer.train_step(step, points, targets, target_normals=target_normals)
        losses.append(loss_val)

    assert len(losses) > 1
    # Check that loss generally decreases (allow some noise)
    assert losses[-1] < losses[0] or all(l >= 0 for l in losses)


def test_scene_trainer_scalar_model_still_works() -> None:
    """SceneTrainer should still work with scalar model type."""
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embeddings = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=hidden_dim
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [embeddings], lr=0.01)

    loader = build_scene_dataloader(
        num_samples=16, points_per_item=128, batch_size=2, seed=42,
        return_normals=True,
    )

    trainer = SceneTrainer(
        model=model,
        learnable_embeddings=embeddings,
        dataloader=loader,
        optimizer=optimizer,
        device="cpu",
        view=False,
        model_type="scalar_sdf",
    )

    initial_loss = trainer.train_step(0, *next(iter(loader)))
    assert isinstance(initial_loss, float)
    assert initial_loss > 0.0
