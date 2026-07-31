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
        view: str | bool | None = None,
        render_every_steps: int = 5,
        render_resolution: int = 128,
    ) -> None:
        self.model = model.to(device)
        self.learnable_embeddings = learnable_embeddings
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.render_every_steps = render_every_steps
        self.criterion = nn.MSELoss()

        view_mode: str | None = None
        if isinstance(view, str):
            view_mode = view if view in ("2d", "3d") else "3d"
        elif isinstance(view, bool) and view:
            view_mode = "3d"

        self.view = view_mode is not None
        self.viewer: LiveSDFViewer | None = None
        if self.view and view_mode is not None:
            self.viewer = LiveSDFViewer(
                title="Scene SDF Live Training",
                resolution=render_resolution,
                view_mode=view_mode,
            )

    def train_step(
        self, step: int, points: torch.Tensor, target_sdf: torch.Tensor
    ) -> float:
        self.model.train()
        points = points.to(self.device)
        target_sdf = target_sdf.to(self.device)
        batch_size = points.shape[0]

        batch_embeddings = self.learnable_embeddings.expand(batch_size, -1, -1)

        # Directional finite difference for Eikonal loss + L1 + MSE surface penalization
        eps = 1e-3
        v = torch.randn_like(points)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        pred_p1 = self.model(points + eps * v, batch_embeddings)
        pred_p2 = self.model(points - eps * v, batch_embeddings)
        gv = (pred_p1 - pred_p2) / (2 * eps)

        eikonal_loss = torch.mean((torch.abs(gv) - 1.0) ** 2)
        pred_sdf = (pred_p1 + pred_p2) / 2.0
        mse_loss = self.criterion(pred_sdf, target_sdf)
        l1_loss = torch.nn.functional.l1_loss(pred_sdf, target_sdf)

        loss = mse_loss + 0.5 * l1_loss + 0.1 * eikonal_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = float(mse_loss.item())

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

    def close(self, keep_open: bool = True) -> None:
        if self.viewer is not None:
            self.viewer.close(keep_open=keep_open)
            self.viewer = None
