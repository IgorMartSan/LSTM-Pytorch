from typing import Optional, Tuple, Any, Dict
import torch

from config import TrainConfig
from models.lstm import LSTMRegressor
from training.checkpointing import load_checkpoint

def load_for_inference(path: str, device: Optional[str] = None):
    ckpt: Dict[str, Any] = load_checkpoint(path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    mean = ckpt["feature_mean"]
    std = ckpt["feature_std"]

    if device is not None:
        cfg.device = device

    model = LSTMRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, mean, std, ckpt
