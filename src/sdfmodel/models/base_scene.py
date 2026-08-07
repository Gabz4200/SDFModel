from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from sdfmodel.models.base import BaseModel
from sdfmodel.models.fourier_pe import FourierPositionEncoding
from sdfmodel.models.transformer_block import CrossAttentionTransformerBlock


class BaseSceneModel(BaseModel):
    """Shared backbone for coordinate-to-embedding implicit scene models.

    Provides Fourier positional encoding, the coordinate MLP stem, and the
    cross-attention transformer stack shared across output heads.
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
        final_activation: Literal["tanh", "sigmoid", "none"] = "none",
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_scene_token = use_scene_token
        self.final_activation = final_activation
        self.out_features = out_features

        self.fourier_pe = FourierPositionEncoding(
            in_features=in_features,
            num_bands=fourier_num_bands,
            learnable=fourier_learnable,
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
        self.dist_head = nn.Linear(hidden_dim, out_features)
        nn.init.normal_(self.dist_head.weight, std=0.01)
        nn.init.zeros_(self.dist_head.bias)

        if use_scene_token:
            self.scene_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

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

    def _forward_features(
        self,
        cords: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if cords.ndim not in (2, 3):
            raise ValueError(
                f"Expected cords to have 2 or 3 dimensions (B, 3) or (B, N, 3), got shape {tuple(cords.shape)}"
            )
        if embedding.ndim not in (2, 3):
            raise ValueError(
                f"Expected embedding to have 2 or 3 dimensions (S, D) or (B, S, D), got shape {tuple(embedding.shape)}"
            )
        if cords.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected coordinate feature dimension {self.in_features}, got {cords.shape[-1]} in shape {tuple(cords.shape)}"
            )
        if embedding.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected embedding hidden dimension {self.hidden_dim}, got {embedding.shape[-1]} in shape {tuple(embedding.shape)}"
            )

        is_2d_emb = embedding.ndim == 2
        if is_2d_emb:
            embedding = embedding.unsqueeze(0)

        is_2d_cords = cords.ndim == 2
        squeeze_dim = None
        if is_2d_cords:
            if embedding.shape[0] > 1 and cords.shape[0] == embedding.shape[0]:
                cords = cords.unsqueeze(1)
                squeeze_dim = 1
            else:
                cords = cords.unsqueeze(0)
                squeeze_dim = 0 if is_2d_emb else None

        if cords.shape[0] != embedding.shape[0]:
            if cords.shape[0] == 1:
                cords = cords.expand(embedding.shape[0], -1, -1)
            elif embedding.shape[0] == 1:
                embedding = embedding.expand(cords.shape[0], -1, -1)
            else:
                raise ValueError(
                    f"Batch size mismatch: cords has batch size {cords.shape[0]} but embedding has batch size {embedding.shape[0]}"
                )

        obj_embed = embedding
        if self.use_scene_token:
            B = obj_embed.shape[0]
            obj_embed = torch.cat(
                [self.scene_token.expand(B, -1, -1), obj_embed], dim=1
            )

        fourier_feats = self.fourier_pe(cords)
        cords_embed = self.cords_mlp(fourier_feats)

        for layer in self.layers:
            cords_embed, obj_embed = layer(cords_embed, obj_embed)

        cords_embed = self.final_norm(cords_embed)
        if squeeze_dim is not None:
            cords_embed = cords_embed.squeeze(squeeze_dim)
        return cords_embed

    def forward(
        self,
        cords: torch.Tensor,
        embedding: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if chunk_size is not None and chunk_size > 0:
            if cords.ndim == 3 and cords.shape[1] > chunk_size:
                chunk_outputs = []
                for i in range(0, cords.shape[1], chunk_size):
                    cords_chunk = cords[:, i : i + chunk_size, :]
                    out_chunk = self.forward(cords_chunk, embedding, chunk_size=None)
                    chunk_outputs.append(out_chunk)
                return torch.cat(chunk_outputs, dim=1)
            elif cords.ndim == 2 and cords.shape[0] > chunk_size:
                chunk_outputs = []
                for i in range(0, cords.shape[0], chunk_size):
                    cords_chunk = cords[i : i + chunk_size, :]
                    out_chunk = self.forward(cords_chunk, embedding, chunk_size=None)
                    chunk_outputs.append(out_chunk)
                return torch.cat(chunk_outputs, dim=0)

        feats = self._forward_features(cords, embedding)
        out = self.dist_head(feats)
        if self.final_activation == "tanh":
            out = torch.tanh(out)
        elif self.final_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out
