import torch
from torch import nn

from sdfmodel.models.cross_attn_sdf import CrossAttnSDFModel


class VectorSDFModel(CrossAttnSDFModel):
    """Implicit Signed Distance Field model returning a 3D vector field.

    The 3D vector at evaluated coordinate p is defined as normal * distance,
    where normal is the unit surface normal pointing outward (grad(SDF))
    and distance is the signed distance to the surface.
    """

    def predict_scalar(
        self,
        cords: torch.Tensor,
        embedding: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Predict scalar Signed Distance Field (SDF) values directly."""
        result = super().forward(cords, embedding, chunk_size=chunk_size)
        # Handle shape: if input was 2D (N, 3) with 3D embedding (1, S, D),
        # parent returns (1, N, 1), squeeze to (N, 1)
        if cords.ndim == 2 and result.ndim == 3 and result.shape[0] == 1:
            result = result.squeeze(0)
        return result

    def _compute_vector_from_scalar(
        self,
        cords: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vector field from scalar SDF by computing gradients."""
        grad_outputs = torch.ones_like(dist)
        grad_cords = torch.autograd.grad(
            outputs=dist,
            inputs=cords,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grad_cords is None:
            grad_cords = torch.zeros_like(cords)

        normals = nn.functional.normalize(grad_cords, dim=-1, eps=1e-8)
        return normals * dist

    def forward(
        self,
        cords: torch.Tensor,
        embedding: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Predict 3D vector field (normal * distance) at given coordinates.

        Args:
            cords: Coordinate tensor of shape (N, 3) or (B, N, 3).
            embedding: Scene object token embedding of shape (S, D) or (B, S, D).
            chunk_size: Optional maximum chunk size for memory-efficient evaluation.

        Returns:
            Vector tensor of shape (N, 3) or (B, N, 3).
        """
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

        # Track original input dimensions
        original_cords_ndim = cords.ndim

        # Ensure cords requires grad for gradient computation
        if not cords.requires_grad:
            cords = cords.detach().requires_grad_(True)

        # Get scalar distance from parent - this handles all the batch/embed logic
        dist = super().forward(cords, embedding, chunk_size=None)
        
        # Compute vector from gradients
        vector = self._compute_vector_from_scalar(cords, dist)
        
        # Handle shape: if original input was 2D (N, 3) with any embedding,
        # parent may have added batch dim -> vector is (1, N, 3), we want (N, 3)
        if original_cords_ndim == 2 and vector.ndim == 3 and vector.shape[0] == 1:
            vector = vector.squeeze(0)
        
        return vector
