import torch
from torch import nn

from sdfmodel.models.base import BaseModel


def compute_eikonal_loss(model: BaseModel, points: torch.Tensor) -> torch.Tensor:
    """Compute Eikonal loss enforcing ||grad(SDF)||_2 = 1."""
    points = points.detach().requires_grad_(True)
    pred_sdf = model(points)

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
    eikonal_loss = torch.mean((grad_norm - 1.0) ** 2)
    return eikonal_loss


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
