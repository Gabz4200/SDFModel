from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sdfmodel.engine.metrics import (
    ScalarModelWrapper,
    compute_combined_sdf_loss,
    compute_vector_sdf_loss,
)
from sdfmodel.models.base import BaseModel
from sdfmodel.render import LiveSDFViewer, create_sdf3_wrapper


class SceneTrainer:
    """Trainer for joint optimization of SDF models and scene object embeddings.

    Supports both CrossAttnSDFModel (scalar SDF output) and VectorSDFModel (3D
    vector field output) with coarse-to-fine warmup, Fourier band annealing, and
    per-term loss logging.
    """

    def __init__(
        self,
        model: BaseModel,
        learnable_embeddings: nn.Parameter,
        dataloader: DataLoader,
        optimizer: Optimizer,
        device: str = "cpu",
        view: str | bool | None = None,
        render_every_steps: int = 5,
        render_resolution: int = 128,
        model_type: Literal["scalar_sdf", "vector_sdf"] = "scalar_sdf",
        w_distance: float = 1.0,
        w_l1: float = 0.5,
        w_eikonal: float = 0.1,
        w_normal: float = 0.2,
        w_vector_l2: float = 1.0,
        w_cosine: float = 0.5,
        w_magnitude_mse: float = 1.0,
        w_consistency: float = 0.0,
        vector_warmup_steps: int = 0,
        total_steps: int | None = None,
        fourier_bands_start: int | None = None,
        fourier_bands_end: int | None = None,
        fourier_anneal_fraction: float = 0.8,
        log_every_steps: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.learnable_embeddings = learnable_embeddings
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.render_every_steps = render_every_steps

        self.w_distance = w_distance
        self.w_l1 = w_l1
        self.w_eikonal = w_eikonal
        self.w_normal = w_normal
        self.w_vector_l2 = w_vector_l2
        self.w_cosine = w_cosine
        self.w_magnitude_mse = w_magnitude_mse
        self.w_consistency = w_consistency
        self.vector_warmup_steps = vector_warmup_steps
        self.log_every_steps = log_every_steps

        pe = getattr(model, "fourier_pe", None)
        if pe is not None and fourier_bands_start is None:
            fourier_bands_start = 4
        self._fourier_bands_start = fourier_bands_start if fourier_bands_start is not None else 0
        self._fourier_bands_end = fourier_bands_end
        if pe is not None and fourier_bands_end is None:
            self._fourier_bands_end = pe.num_bands
        self._fourier_anneal_fraction = fourier_anneal_fraction
        self._total_steps = total_steps

        view_mode: str | None = None
        if isinstance(view, str):
            view_mode = view if view in ("2d", "3d") else "3d"
        elif isinstance(view, bool) and view:
            view_mode = "3d"

        self.view = view_mode is not None
        self.viewer: LiveSDFViewer | None = None
        if self.view and view_mode is not None:
            original_sdf = getattr(dataloader.dataset, "scene", None)
            self.viewer = LiveSDFViewer(
                title="Scene SDF Live Training",
                resolution=render_resolution,
                view_mode=view_mode,
                original_sdf=original_sdf,
            )

    def _apply_fourier_anneal(self, step: int) -> None:
        if self._total_steps is None or self._fourier_bands_start == 0:
            return
        if self._fourier_bands_end is None or self._fourier_bands_start >= self._fourier_bands_end:
            return
        pe = getattr(self.model, "fourier_pe", None)
        if pe is None:
            return
        progress = min(1.0, (step + 1) / (self._total_steps * self._fourier_anneal_fraction))
        active = self._fourier_bands_start + progress * (self._fourier_bands_end - self._fourier_bands_start)
        pe.active_bands = active

    def _compute_loss(
        self,
        step: int,
        points: torch.Tensor,
        target_sdf: torch.Tensor,
        target_normals: torch.Tensor | None,
        batch_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scalar_warmup = (
            self.model_type == "vector_sdf"
            and step < self.vector_warmup_steps
        )

        if scalar_warmup:
            scalar_model = ScalarModelWrapper(self.model)
            return compute_combined_sdf_loss(
                model=scalar_model,
                points=points,
                target_sdf=target_sdf,
                target_normals=target_normals,
                embedding=batch_embeddings,
                w_distance=self.w_distance,
                w_l1=self.w_l1,
                w_eikonal=self.w_eikonal,
                w_normal=self.w_normal if target_normals is not None else 0.0,
            )

        if self.model_type == "vector_sdf":
            return compute_vector_sdf_loss(
                model=self.model,
                points=points,
                target_sdf=target_sdf,
                target_normals=target_normals,
                embedding=batch_embeddings,
                w_vector_l2=self.w_vector_l2,
                w_cosine=self.w_cosine,
                w_magnitude_mse=self.w_magnitude_mse,
                w_eikonal=self.w_eikonal,
                w_normal=self.w_normal if target_normals is not None else 0.0,
                w_consistency=self.w_consistency,
            )

        return compute_combined_sdf_loss(
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

        self._apply_fourier_anneal(step)

        loss_dict = self._compute_loss(
            step, points, target_sdf, target_normals, batch_embeddings
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

        if (
            self.log_every_steps > 0
            and step % self.log_every_steps == 0
        ):
            parts = []
            for k, v in loss_dict.items():
                if k == "loss":
                    continue
                parts.append(f"{k}={float(v.detach()):.6f}")
            if parts:
                print(f"[step {step}] " + " | ".join(parts))

        return loss_val

    def fit(self, epochs: int) -> dict[str, float]:
        """Run the training loop for the given number of epochs."""
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, (points, target_sdf, *rest) in enumerate(
                self.dataloader, start=1
            ):
                target_normals = rest[0] if rest else None
                epoch_loss += self.train_step(
                    step, points, target_sdf, target_normals
                )
            avg = epoch_loss / max(len(self.dataloader), 1)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.6f}")
            epoch_losses.append(avg)

        self.close(keep_open=True)
        return {"final_loss": epoch_losses[-1] if epoch_losses else float("inf")}

    def close(self, keep_open: bool = True) -> None:
        if self.viewer is not None:
            self.viewer.close(keep_open=keep_open)
            self.viewer = None
