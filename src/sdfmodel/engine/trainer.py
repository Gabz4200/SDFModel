from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from sdfmodel.engine.metrics import compute_sdf_metrics
from sdfmodel.models.base import BaseModel
from sdfmodel.utils.config import ExperimentConfig


class Trainer:
    """Device-agnostic PyTorch research trainer with evaluation loop and checkpointing."""

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.training.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6,
        )

        self.criterion = nn.MSELoss()
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for points, target_sdf in self.train_loader:
            points = points.to(self.device)
            target_sdf = target_sdf.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                pred_sdf = self.model(points)
                loss = self.criterion(pred_sdf, target_sdf)

            if self.use_amp:
                scaled_loss = cast(torch.Tensor, self.scaler.scale(loss))
                scaled_loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * points.size(0)

        self.scheduler.step()
        dataset = cast(Sized, self.train_loader.dataset)
        return total_loss / len(dataset)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []

        for points, target_sdf in self.val_loader:
            points = points.to(self.device)
            pred_sdf = self.model(points)
            all_preds.append(pred_sdf.cpu())
            all_targets.append(target_sdf.cpu())

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        return compute_sdf_metrics(cat_preds, cat_targets)

    def fit(self) -> dict[str, float]:
        best_val_mse = float("inf")

        for _ in range(1, self.config.training.epochs + 1):
            _ = self.train_epoch()
            val_metrics = self.evaluate()

            if val_metrics["mse"] < best_val_mse:
                best_val_mse = val_metrics["mse"]
                self.save_checkpoint("best.pt")

        return self.evaluate()

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> Path:
        path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
