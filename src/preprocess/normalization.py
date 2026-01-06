import torch
from typing import Tuple

def compute_train_stats(series: torch.Tensor, train_ratio: float) -> Tuple[float, float]:
    train_size = max(int(series.size(0) * train_ratio), 1)
    train_slice = series[:train_size]
    mean = train_slice.mean().item()
    std = train_slice.std(unbiased=False).item()
    std = std if std > 1e-12 else 1.0
    return mean, std

def normalize_with_stats(series: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    safe_std = std if std > 1e-12 else 1.0
    return (series - mean) / safe_std

def denormalize(series_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return series_norm * std + mean
