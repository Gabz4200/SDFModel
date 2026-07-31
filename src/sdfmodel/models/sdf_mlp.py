import torch
from torch import nn

from sdfmodel.models.base import BaseModel
from sdfmodel.models.fourier_pe import FourierPositionEncoding


class SDFMLP(BaseModel):
    """Implicit neural representation MLP for Signed Distance Functions (SDF)."""

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        num_layers: int = 4,
        out_features: int = 1,
        use_fourier_pe: bool = True,
        fourier_num_bands: int = 6,
    ) -> None:
        super().__init__()

        self.use_fourier_pe = use_fourier_pe
        if use_fourier_pe:
            self.pe: FourierPositionEncoding | None = FourierPositionEncoding(
                in_features=in_features, num_bands=fourier_num_bands
            )
            input_dim = self.pe.out_features
        else:
            self.pe = None
            input_dim = in_features

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_features))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe is not None:
            x = self.pe(x)
        return self.net(x)
