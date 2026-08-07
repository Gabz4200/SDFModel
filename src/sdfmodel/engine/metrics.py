import torch
from torch import nn

from sdfmodel.models.vector_sdf import VectorSDFModel


def _eval_model(
    model: nn.Module, points: torch.Tensor, embedding: torch.Tensor | None = None
) -> torch.Tensor:
    if embedding is not None:
        if (
            embedding.ndim == 3
            and points.ndim == 3
            and embedding.shape[0] != points.shape[0]
            and embedding.shape[0] == 1
        ):
            embedding = embedding.expand(points.shape[0], -1, -1)
        return model(points, embedding)
    return model(points)


def _scalar_wrapper(
    model: nn.Module, points: torch.Tensor, embedding: torch.Tensor | None = None
) -> torch.Tensor:
    """Evaluate model returning scalar SDF values, supporting VectorSDFModel and standard models."""
    if isinstance(model, VectorSDFModel):
        if embedding is not None:
            return model.predict_scalar(points, embedding)
        raise ValueError("VectorSDFModel requires an embedding parameter")
    if embedding is not None:
        return model(points, embedding)
    return model(points)


class ScalarModelWrapper(nn.Module):
    """Wrap a VectorSDFModel (or any model) to expose scalar SDF output."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(
        self, pts: torch.Tensor, emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        return _scalar_wrapper(self.m, pts, emb)


def compute_eikonal_loss(
    model: nn.Module,
    points: torch.Tensor,
    embedding: torch.Tensor | None = None,
    use_autograd: bool = False,
    eps_min: float = 1e-4,
    eps_max: float = 1e-2,
    alpha: float = 1e-2,
) -> torch.Tensor:
    """Compute Eikonal loss enforcing ||grad(SDF)||_2 = 1.

    Supports exact autograd gradients or directional finite difference gradients
    with adaptive epsilon scaling: eps(p) = clamp(alpha * |f(p)|, eps_min, eps_max).
    """
    if use_autograd:
        points = points.detach().requires_grad_(True)
        pred_sdf = _eval_model(model, points, embedding)

        grad_outputs = torch.ones_like(pred_sdf)
        gradients = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = gradients.norm(dim=-1)
        return torch.mean((grad_norm - 1.0) ** 2)

    pred_sdf = _eval_model(model, points, embedding)
    abs_f = torch.abs(pred_sdf.detach())
    eps = torch.clamp(alpha * abs_f, min=eps_min, max=eps_max)

    v = torch.randn_like(points)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    pred_p1 = _eval_model(model, points + eps * v, embedding)
    pred_p2 = _eval_model(model, points - eps * v, embedding)

    gv = (pred_p1 - pred_p2) / (2.0 * eps)
    return torch.mean((torch.abs(gv) - 1.0) ** 2)


def compute_sdf_normals(
    model: nn.Module,
    points: torch.Tensor,
    embedding: torch.Tensor | None = None,
    method: str = "central",
    h: float = 1e-4,
) -> torch.Tensor:
    """Compute surface unit normal vectors for model predictions at points.

    Supports 'central' (6-eval difference), 'tetrahedron' (4-eval IQ technique),
    or 'autograd' (exact gradients).
    """
    if method == "autograd":
        pts = points.detach().requires_grad_(True)
        pred_sdf = _eval_model(model, pts, embedding)
        grad_outputs = torch.ones_like(pred_sdf)
        gradients = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=pts,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return nn.functional.normalize(gradients, dim=-1, eps=1e-8)

    elif method == "tetrahedron":
        k0 = torch.tensor([1.0, -1.0, -1.0], device=points.device, dtype=points.dtype)
        k1 = torch.tensor([-1.0, -1.0, 1.0], device=points.device, dtype=points.dtype)
        k2 = torch.tensor([-1.0, 1.0, -1.0], device=points.device, dtype=points.dtype)
        k3 = torch.tensor([1.0, 1.0, 1.0], device=points.device, dtype=points.dtype)

        f0 = _eval_model(model, points + h * k0, embedding)
        f1 = _eval_model(model, points + h * k1, embedding)
        f2 = _eval_model(model, points + h * k2, embedding)
        f3 = _eval_model(model, points + h * k3, embedding)

        g = (k0 * f0 + k1 * f1 + k2 * f2 + k3 * f3) / (4.0 * h)
        return nn.functional.normalize(g, dim=-1, eps=1e-8)

    elif method == "central":
        ex = torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype)
        ey = torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype)
        ez = torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype)

        fx1 = _eval_model(model, points + h * ex, embedding)
        fx2 = _eval_model(model, points - h * ex, embedding)
        fy1 = _eval_model(model, points + h * ey, embedding)
        fy2 = _eval_model(model, points - h * ey, embedding)
        fz1 = _eval_model(model, points + h * ez, embedding)
        fz2 = _eval_model(model, points - h * ez, embedding)

        dx = (fx1 - fx2) / (2.0 * h)
        dy = (fy1 - fy2) / (2.0 * h)
        dz = (fz1 - fz2) / (2.0 * h)

        g = torch.cat([dx, dy, dz], dim=-1)
        return nn.functional.normalize(g, dim=-1, eps=1e-8)
    else:
        raise ValueError(
            f"Unknown normal computation method: {method}. Choose from 'central', 'tetrahedron', or 'autograd'."
        )


def compute_normal_loss(
    model: nn.Module,
    points: torch.Tensor,
    target_normals: torch.Tensor,
    embedding: torch.Tensor | None = None,
    method: str = "central",
    h: float = 1e-4,
) -> torch.Tensor:
    """Compute normal vector loss using 1 - cosine_similarity(pred_normals, target_normals)."""
    pred_normals = compute_sdf_normals(
        model, points, embedding=embedding, method=method, h=h
    )
    cos_sim = nn.functional.cosine_similarity(pred_normals, target_normals, dim=-1)
    return torch.mean(1.0 - cos_sim)


def compute_combined_sdf_loss(
    model: nn.Module,
    points: torch.Tensor,
    target_sdf: torch.Tensor,
    target_normals: torch.Tensor | None = None,
    embedding: torch.Tensor | None = None,
    w_distance: float = 1.0,
    w_l1: float = 0.5,
    w_eikonal: float = 0.1,
    w_normal: float = 0.2,
    normal_method: str = "central",
    use_autograd_eikonal: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute combined multi-term SDF loss integrating distance, gradient (Eikonal), and normal vector losses."""
    pred_sdf = _eval_model(model, points, embedding)
    mse_loss = nn.functional.mse_loss(pred_sdf, target_sdf)
    l1_loss = nn.functional.l1_loss(pred_sdf, target_sdf)

    eikonal_loss = compute_eikonal_loss(
        model, points, embedding=embedding, use_autograd=use_autograd_eikonal
    )

    if target_normals is not None and w_normal > 0.0:
        normal_loss = compute_normal_loss(
            model,
            points,
            target_normals,
            embedding=embedding,
            method=normal_method,
        )
    else:
        normal_loss = torch.tensor(0.0, device=points.device, dtype=points.dtype)

    total_loss = (
        w_distance * mse_loss
        + w_l1 * l1_loss
        + w_eikonal * eikonal_loss
        + w_normal * normal_loss
    )

    return {
        "loss": total_loss,
        "mse_loss": mse_loss,
        "l1_loss": l1_loss,
        "eikonal_loss": eikonal_loss,
        "normal_loss": normal_loss,
    }


def compute_vector_sdf_loss(
    model: nn.Module,
    points: torch.Tensor,
    target_sdf: torch.Tensor,
    target_normals: torch.Tensor | None = None,
    embedding: torch.Tensor | None = None,
    w_vector_l2: float = 1.0,
    w_cosine: float = 0.5,
    w_magnitude_mse: float = 1.0,
    w_eikonal: float = 0.1,
    w_normal: float = 0.2,
    w_consistency: float = 0.0,
    normal_method: str = "central",
    use_autograd_eikonal: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute vector-specific SDF training loss.

    Target vector is ``target_sdf * target_normals``.
    With a learned direction head the vector field satisfies ``|v| == |f|`` by
    construction; an optional consistency term aligns the learned direction
    with the finite-difference gradient of the scalar field.
    """
    pred_vector = _eval_model(model, points, embedding)
    scalar_model = ScalarModelWrapper(model)

    if target_normals is None:
        target_normals = compute_sdf_normals(
            scalar_model, points, embedding=embedding, method=normal_method
        )
        pred_sdf_scalar = _scalar_wrapper(model, points, embedding)
        target_vector = pred_sdf_scalar * target_normals
    else:
        target_vector = target_sdf * target_normals

    vector_l2_loss = nn.functional.mse_loss(pred_vector, target_vector)

    pred_norm_vec = nn.functional.normalize(pred_vector, dim=-1, eps=1e-8)
    target_norm_vec = nn.functional.normalize(target_vector, dim=-1, eps=1e-8)
    cos_sim = nn.functional.cosine_similarity(pred_norm_vec, target_norm_vec, dim=-1)
    cosine_loss = torch.mean(1.0 - cos_sim)

    pred_mag = pred_vector.norm(dim=-1, keepdim=True)
    target_mag = target_sdf.abs()
    magnitude_mse_loss = nn.functional.mse_loss(pred_mag, target_mag)

    eikonal_loss = compute_eikonal_loss(
        scalar_model, points, embedding=embedding, use_autograd=use_autograd_eikonal
    )

    normal_loss = torch.tensor(0.0, device=points.device, dtype=points.dtype)
    consistency_loss = torch.tensor(0.0, device=points.device, dtype=points.dtype)
    if w_normal > 0.0 or w_consistency > 0.0:
        fd_normals = compute_sdf_normals(
            scalar_model, points, embedding=embedding, method=normal_method
        )
        if w_normal > 0.0:
            normal_loss = torch.mean(
                1.0
                - nn.functional.cosine_similarity(fd_normals, target_normals, dim=-1)
            )
        if w_consistency > 0.0:
            pred_dir = nn.functional.normalize(pred_vector, dim=-1, eps=1e-8)
            consistency_loss = torch.mean(
                1.0 - nn.functional.cosine_similarity(pred_dir, fd_normals, dim=-1)
            )

    total_loss = (
        w_vector_l2 * vector_l2_loss
        + w_cosine * cosine_loss
        + w_magnitude_mse * magnitude_mse_loss
        + w_eikonal * eikonal_loss
        + w_normal * normal_loss
        + w_consistency * consistency_loss
    )

    return {
        "loss": total_loss,
        "vector_l2_loss": vector_l2_loss,
        "cosine_loss": cosine_loss,
        "magnitude_mse_loss": magnitude_mse_loss,
        "eikonal_loss": eikonal_loss,
        "normal_loss": normal_loss,
        "consistency_loss": consistency_loss,
    }


def compute_sdf_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute MSE, L1, and Peak Signal-to-Noise Ratio (PSNR) metrics for SDF prediction."""
    mse = nn.functional.mse_loss(pred, target).item()
    l1 = nn.functional.l1_loss(pred, target).item()

    target_range = max(target.max().item() - target.min().item(), 1e-6)
    if mse > 0:
        psnr = 20.0 * torch.log10(torch.tensor(target_range)) - 10.0 * torch.log10(
            torch.tensor(mse)
        )
        psnr_val = psnr.item()
    else:
        psnr_val = float("inf")

    return {"mse": mse, "l1": l1, "psnr": psnr_val}
