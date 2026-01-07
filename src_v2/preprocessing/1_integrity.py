from typing import Tuple
import torch


def clean_series(series: torch.Tensor) -> Tuple[torch.Tensor, int]:
    series = series.flatten()
    mask = torch.isfinite(series)
    cleaned = series[mask]
    removed = int(series.numel() - cleaned.numel())
    if cleaned.numel() == 0:
        raise ValueError("Série vazia após remover valores inválidos.")
    return cleaned, removed
