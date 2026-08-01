import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sdfmodel.engine.metrics import compute_combined_sdf_loss
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
        w_distance: float = 1.0,
        w_l1: float = 0.5,
        w_eikonal: float = 0.1,
        w_normal: float = 0.2,
    ) -> None:
        self.model = model.to(device)
        self.learnable_embeddings = learnable_embeddings
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.render_every_steps = render_every_steps
        self.w_distance = w_distance
        self.w_l1 = w_l1
        self.w_eikonal = w_eikonal
        self.w_normal = w_normal
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
        self,
        step: int,
        points: torch.Tensor,
        target_sdf: torch.Tensor,
        target_normals: torch.Tensor | None = None,
    ) -> float:
        self.model.train()
        points = points.to(self.device)
        target_sdf = target_sdf.to(self.device)
        if target_normals is not None:
            target_normals = target_normals.to(self.device)
        batch_size = points.shape[0]

        batch_embeddings = self.learnable_embeddings.expand(batch_size, -1, -1)

        loss_dict = compute_combined_sdf_loss(
            model=self.model,
            points=points,
            target_sdf=target_sdf,
            target_normals=target_normals,
            embedding=batch_embeddings,
            w_distance=self.w_distance,
            w_l1=self.w_l1,
            w_eikonal=self.w_eikonal,
            w_normal=self.w_normal if target_normals is not None else 0.0,
        )

        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = float(loss_dict["loss"].item())

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
            self.viewer.update(sdf_obj, step=step, loss=loss_dict)

        return loss_val

    def close(self, keep_open: bool = True) -> None:
        if self.viewer is not None:
            self.viewer.close(keep_open=keep_open)
            self.viewer = None
