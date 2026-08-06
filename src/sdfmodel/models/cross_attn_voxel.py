from sdfmodel.models.base_scene import BaseSceneModel


class CrossAttnVoxelModel(BaseSceneModel):
    """Implicit Voxel model using Fourier features and Cross-Attention over scene object embeddings.

    For each sampled coordinate, outputs a 4D vector (exist, red, green, blue):
      - `exist`: sigmoid activation, approaches 1.0 when a voxel is present.
      - `red`, `green`, `blue`: sigmoid activation, represent color in [0, 1].
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        fourier_num_bands: int = 6,
        fourier_learnable: bool = False,
        dropout: float = 0.0,
        use_scene_token: bool = False,
    ) -> None:
        super().__init__(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            fourier_num_bands=fourier_num_bands,
            fourier_learnable=fourier_learnable,
            dropout=dropout,
            use_scene_token=use_scene_token,
            final_activation="sigmoid",
            out_features=4,
        )
