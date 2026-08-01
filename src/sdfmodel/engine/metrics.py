import torch
from torch import nn


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
