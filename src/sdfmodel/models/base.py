from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Base PyTorch module for research models providing utility helpers."""

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass shape contract must be enforced in subclasses."""
