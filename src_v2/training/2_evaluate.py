from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader

from src_v2.loader import load_module

_norm = load_module(__file__, "../preprocessing/2_normalization.py", "norm")


def _denormalize(series_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return _norm.denormalize(series_norm, mean, std)


@torch.no_grad()
def evaluate_denorm_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        preds.append(out.cpu())
        trues.append(y.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    preds_denorm = _denormalize(preds, mean, std)
    trues_denorm = _denormalize(trues, mean, std)

    mae = torch.mean(torch.abs(preds_denorm - trues_denorm)).item()
    rmse = torch.sqrt(torch.mean((preds_denorm - trues_denorm) ** 2)).item()
    mape = (torch.mean(torch.abs((trues_denorm - preds_denorm) / trues_denorm.clamp(min=1e-3))) * 100).item()

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


@torch.no_grad()
def compute_val_loss_mse(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    running = 0.0
    n = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        running += loss.item() * x.size(0)
        n += x.size(0)

    return running / max(n, 1)
