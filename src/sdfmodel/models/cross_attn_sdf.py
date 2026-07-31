import torch
from torch import nn

from sdfmodel.models.base import BaseModel
from sdfmodel.models.fourier_pe import FourierPositionEncoding


class CrossAttentionTransformerBlock(nn.Module):
    """Transformer block with self-attention for scene object tokens and cross-attention from coordinates."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Self-attention for scene object embedding sequence
        self.obj_norm1 = nn.LayerNorm(hidden_dim)
        self.obj_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.obj_norm2 = nn.LayerNorm(hidden_dim)
        ffn_hidden_dim = int(hidden_dim * ffn_ratio)
        self.obj_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Cross-attention: coordinates (queries) -> object embeddings (keys, values)
        self.cords_norm1 = nn.LayerNorm(hidden_dim)
        self.cords_cross_attn_kv_norm = nn.LayerNorm(hidden_dim)
        self.cords_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN for coordinate representations
        self.cords_norm2 = nn.LayerNorm(hidden_dim)
        self.cords_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cords_embed: torch.Tensor,
        obj_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Object-object interaction via self-attention
        norm_obj = self.obj_norm1(obj_embed)
        obj_attn_out, _ = self.obj_self_attn(
            query=norm_obj, key=norm_obj, value=norm_obj, need_weights=False
        )
        obj_embed = obj_embed + obj_attn_out
        obj_embed = obj_embed + self.obj_ffn(self.obj_norm2(obj_embed))

        # 2. Coordinate queries cross-attending to object sequence
        norm_cords = self.cords_norm1(cords_embed)
        norm_kv_obj = self.cords_cross_attn_kv_norm(obj_embed)
        cords_attn_out, _ = self.cords_cross_attn(
            query=norm_cords,
            key=norm_kv_obj,
            value=norm_kv_obj,
            need_weights=False,
        )
        cords_embed = cords_embed + cords_attn_out

        # 3. Coordinate representation update via GELU FFN
        cords_embed = cords_embed + self.cords_ffn(self.cords_norm2(cords_embed))

        return cords_embed, obj_embed


class CrossAttnSDFModel(BaseModel):
    """Implicit Signed Distance Field model using Fourier features and Cross-Attention over scene object embeddings."""

    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        fourier_num_bands: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fourier_pe = FourierPositionEncoding(
            in_features=in_features, num_bands=fourier_num_bands
        )
        self.cords_mlp = nn.Sequential(
            nn.Linear(self.fourier_pe.out_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                CrossAttentionTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dist_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def create_learnable_embedding(
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        """Helper to construct an external learnable parameter tensor for object embeddings."""
        param = torch.empty(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
        nn.init.normal_(param, std=0.02)
        return nn.Parameter(param)

    def forward(self, cords: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if cords.ndim not in (2, 3):
            raise ValueError(
                f"Expected cords to have 2 or 3 dimensions (B, 3) or (B, N, 3), got shape {tuple(cords.shape)}"
            )
        if embedding.ndim != 3:
            raise ValueError(
                f"Expected embedding to have 3 dimensions (B, S, D), got shape {tuple(embedding.shape)}"
            )

        if cords.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"Batch size mismatch: cords has batch size {cords.shape[0]} but embedding has batch size {embedding.shape[0]}"
            )

        if cords.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected coordinate feature dimension {self.in_features}, got {cords.shape[-1]} in shape {tuple(cords.shape)}"
            )

        if embedding.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected embedding hidden dimension {self.hidden_dim}, got {embedding.shape[-1]} in shape {tuple(embedding.shape)}"
            )

        is_2d = cords.ndim == 2
        if is_2d:
            cords = cords.unsqueeze(1)

        fourier_feats = self.fourier_pe(cords)
        cords_embed = self.cords_mlp(fourier_feats)

        obj_embed = embedding
        for layer in self.layers:
            cords_embed, obj_embed = layer(cords_embed, obj_embed)

        cords_embed = self.final_norm(cords_embed)
        dist = self.dist_head(cords_embed)

        if is_2d:
            dist = dist.squeeze(1)

        return dist
