import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel
from sdfmodel.render import LiveSDFViewer, create_sdf3_wrapper


class SceneTrainer:
    """Trainer for joint optimization of CrossAttnSDFModel and 4-primitive scene object embeddings."""

    def __init__(
        self,
        model: CrossAttnSDFModel,
        learnable_embeddings: nn.Parameter,
        dataloader: DataLoader,
        optimizer: Optimizer,
        device: str = "cpu",
        view: bool = False,
        render_every_steps: int = 5,
        render_resolution: int = 128,
    ) -> None:
        self.model = model.to(device)
        self.learnable_embeddings = learnable_embeddings
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.view = view
        self.render_every_steps = render_every_steps
        self.criterion = nn.MSELoss()

        self.viewer: LiveSDFViewer | None = None
        if self.view:
            self.viewer = LiveSDFViewer(
                title="Scene SDF Live Training", resolution=render_resolution
            )

    def train_step(
        self, step: int, points: torch.Tensor, target_sdf: torch.Tensor
    ) -> float:
        self.model.train()
        points = points.to(self.device)
        target_sdf = target_sdf.to(self.device)
        batch_size = points.shape[0]

        # Expand (1, 4, hidden_dim) embeddings to (batch_size, 4, hidden_dim)
        batch_embeddings = self.learnable_embeddings.expand(batch_size, -1, -1)

        self.optimizer.zero_grad()
        pred_sdf = self.model(points, batch_embeddings)
        loss = self.criterion(pred_sdf, target_sdf)
        loss.backward()
        self.optimizer.step()

        loss_val = float(loss.item())

        if (
            self.view
            and self.viewer is not None
            and (step % self.render_every_steps == 0)
        ):
            sdf_obj = create_sdf3_wrapper(
                self.model,
                embedding=self.learnable_embeddings.detach(),
                device=self.device,
            )
            self.viewer.update(sdf_obj, step=step, loss=loss_val)

        return loss_val

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
