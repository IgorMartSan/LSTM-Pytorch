from typing import Any, Dict, Optional, Tuple
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src_v2.config import TrainConfig
from src_v2.loader import load_module

_model = load_module(__file__, "../model/1_lstm.py", "lstm_model")
_ckpt = load_module(__file__, "3_checkpointing.py", "checkpointing")
_norm = load_module(__file__, "../preprocessing/2_normalization.py", "norm")
_integrity = load_module(__file__, "../preprocessing/1_integrity.py", "integrity")


def load_for_inference(path: str, device: Optional[str] = None):
    ckpt: Dict[str, Any] = _ckpt.load_checkpoint(path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    mean = ckpt["feature_mean"]
    std = ckpt["feature_std"]

    if device is not None:
        cfg.device = device

    model = _model.LSTMRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, mean, std, ckpt


def predict_next_from_series(
    series: torch.Tensor,
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    model, cfg, mean, std, ckpt = load_for_inference(checkpoint_path, device=device)

    cleaned, _ = _integrity.clean_series(series)
    if cleaned.numel() < cfg.sequence_length:
        raise ValueError("Serie curta para a janela configurada no checkpoint.")

    window = cleaned[-cfg.sequence_length :].unsqueeze(0).unsqueeze(-1)
    window_norm = _norm.normalize_with_stats(window, mean, std)

    device_obj = torch.device(cfg.device)
    model = model.to(device_obj)
    window_norm = window_norm.to(device_obj)

    with torch.no_grad():
        pred_norm = model(window_norm).cpu()

    pred = _norm.denormalize(pred_norm, mean, std)
    return float(pred.squeeze().item()), ckpt
