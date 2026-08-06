import torch
from torch import nn

from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


class VectorSDFModel(CrossAttnSDFModel):
    """Implicit Signed Distance Field model returning a 3D vector field.

    The 3D vector at evaluated coordinate p is defined as ``normal * distance``,
    where ``normal`` is a learned unit-surface-normal direction and
    ``distance`` is the scalar SDF value. This guarantees ``|v| == |f|`` and
    keeps the scalar and vector fields consistent by construction.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vector_head = nn.Linear(self.hidden_dim, self.in_features)
        nn.init.normal_(self.vector_head.weight, std=0.01)
        nn.init.zeros_(self.vector_head.bias)

    def predict_scalar(
        self,
        cords: torch.Tensor,
        embedding: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Predict scalar Signed Distance Field (SDF) values directly."""
        if chunk_size is not None and chunk_size > 0:
            if cords.ndim == 3 and cords.shape[1] > chunk_size:
                chunk_outputs = []
                for i in range(0, cords.shape[1], chunk_size):
                    out_chunk = self.predict_scalar(
                        cords[:, i : i + chunk_size, :], embedding, chunk_size=None
                    )
                    chunk_outputs.append(out_chunk)
                return torch.cat(chunk_outputs, dim=1)
            elif cords.ndim == 2 and cords.shape[0] > chunk_size:
                chunk_outputs = []
                for i in range(0, cords.shape[0], chunk_size):
                    out_chunk = self.predict_scalar(
                        cords[i : i + chunk_size, :], embedding, chunk_size=None
                    )
                    chunk_outputs.append(out_chunk)
                return torch.cat(chunk_outputs, dim=0)

        feats = self._forward_features(cords, embedding)
        dist = self.dist_head(feats)
        if self.use_tanh:
            dist = torch.tanh(dist)
        if cords.ndim == 2 and dist.ndim == 3 and dist.shape[0] == 1:
            dist = dist.squeeze(0)
        return dist

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
        dist = self.dist_head(feats)
        if self.use_tanh:
            dist = torch.tanh(dist)
        direction = nn.functional.normalize(
            self.vector_head(feats), dim=-1, eps=1e-8
        )
        vector = direction * dist

        original_cords_ndim = cords.ndim
        if original_cords_ndim == 2 and vector.ndim == 3 and vector.shape[0] == 1:
            vector = vector.squeeze(0)

        return vector
