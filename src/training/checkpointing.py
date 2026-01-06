import os
import torch
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def trial_ckpt_path(checkpoints_dir: str, trial_number: int) -> str:
    ensure_dir(checkpoints_dir)
    return os.path.join(checkpoints_dir, f"trial_{trial_number}.pt")

def save_checkpoint(path: str, checkpoint: Dict[str, Any]) -> None:
    torch.save(checkpoint, path)

def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
