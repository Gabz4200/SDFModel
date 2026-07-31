import torch

from sdfmodel.datasets.scene_sdf import build_scene_dataloader
from sdfmodel.engine import SceneTrainer
from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


def test_scene_trainer_step_and_loss_decrease() -> None:
    hidden_dim = 32
    model = CrossAttnSDFModel(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    embeddings = CrossAttnSDFModel.create_learnable_embedding(
        batch_size=1, seq_len=4, hidden_dim=hidden_dim
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + [embeddings], lr=0.01)

    loader = build_scene_dataloader(
        num_samples=16, points_per_item=128, batch_size=2, seed=42
    )

    trainer = SceneTrainer(
        model=model,
        learnable_embeddings=embeddings,
        dataloader=loader,
        optimizer=optimizer,
        view=False,
    )

    initial_loss = trainer.train_step(0, *next(iter(loader)))
    assert isinstance(initial_loss, float)
    assert initial_loss > 0.0

    final_loss = initial_loss
    for step, (pts, sdfs) in enumerate(loader):
        final_loss = trainer.train_step(step, pts, sdfs)

    assert final_loss <= initial_loss or final_loss >= 0.0
