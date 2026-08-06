import math

import torch
from torch import nn


class FourierPositionEncoding(nn.Module):
    """Fourier feature positional encoding for coordinate inputs."""

    frequencies: torch.Tensor

    def __init__(
        self, in_features: int = 3, num_bands: int = 6, learnable: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_bands = num_bands
        self.learnable = learnable

        freqs = torch.pow(2.0, torch.arange(num_bands, dtype=torch.float32)) * math.pi
        if learnable:
            self.frequencies = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer("frequencies", freqs, persistent=True)

        self.active_bands: int | None = None

    @property
    def out_features(self) -> int:
        return self.in_features * (1 + 2 * self.num_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1) * self.frequencies
        sin_feat = torch.sin(x_expanded).flatten(start_dim=-2)
        cos_feat = torch.cos(x_expanded).flatten(start_dim=-2)

        if self.active_bands is not None and self.active_bands < self.num_bands:
            band_idx = torch.arange(self.num_bands, device=x.device, dtype=x.dtype)
            weights = torch.clamp(self.active_bands - band_idx, 0.0, 1.0)
            weights = weights.repeat(self.in_features)
            sin_feat = sin_feat * weights
            cos_feat = cos_feat * weights

        return torch.cat([x, sin_feat, cos_feat], dim=-1)
