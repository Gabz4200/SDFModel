import torch
from torch import nn


class CrossAttentionTransformerBlock(nn.Module):
    """Transformer block with cross-attention from coordinates to scene object tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cords_norm1 = nn.LayerNorm(hidden_dim)
        self.cords_cross_attn_kv_norm = nn.LayerNorm(hidden_dim)
        self.cords_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        ffn_hidden_dim = int(hidden_dim * ffn_ratio)
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
        norm_cords = self.cords_norm1(cords_embed)
        norm_kv_obj = self.cords_cross_attn_kv_norm(obj_embed)
        cords_attn_out, _ = self.cords_cross_attn(
            query=norm_cords,
            key=norm_kv_obj,
            value=norm_kv_obj,
            need_weights=False,
        )
        cords_embed = cords_embed + cords_attn_out
        cords_embed = cords_embed + self.cords_ffn(self.cords_norm2(cords_embed))
        return cords_embed, obj_embed
