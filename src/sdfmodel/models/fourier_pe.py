import math

import torch
from torch import nn


class FourierPositionEncoding(nn.Module):
    """Fourier feature positional encoding for coordinate inputs."""

    frequencies: torch.Tensor

    def __init__(self, in_features: int = 3, num_bands: int = 6) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_bands = num_bands

        freqs = torch.pow(2.0, torch.arange(num_bands, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", freqs, persistent=True)

    @property
    def out_features(self) -> int:
        return self.in_features * (1 + 2 * self.num_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1) * self.frequencies
        sin_feat = torch.sin(x_expanded).flatten(start_dim=-2)
        cos_feat = torch.cos(x_expanded).flatten(start_dim=-2)

        return torch.cat([x, sin_feat, cos_feat], dim=-1)
