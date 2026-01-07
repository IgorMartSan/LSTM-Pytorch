from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src_v2.config import TrainConfig
from src_v2.loader import load_module

_model = load_module(__file__, "../model/1_lstm.py", "lstm_model")
_eval = load_module(__file__, "2_evaluate.py", "evaluate")
_ckpt = load_module(__file__, "3_checkpointing.py", "checkpointing")
_window = load_module(__file__, "../preprocessing/3_windowing.py", "windowing")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    cfg: TrainConfig,
    normalized_series: torch.Tensor,
    feature_mean: float,
    feature_std: float,
    checkpoint_path: Optional[str] = None,
    trial_number: Optional[int] = None,
    trial_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], str, float]:
    device = torch.device(cfg.device)

    inputs, targets = _window.create_window_tensors(normalized_series, cfg.sequence_length)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _window.split_train_val_test(
        inputs, targets, cfg.train_ratio, cfg.val_ratio_within_train
    )

    train_ds = TensorDataset(tr_x, tr_y)
    val_ds = TensorDataset(va_x, va_y)
    test_ds = TensorDataset(te_x, te_y)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = _model.LSTMRegressor(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(1, cfg.max_epochs + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss = _eval.compute_val_loss_mse(model, val_loader, device)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Falha: best_state nao foi definido.")

    model.load_state_dict(best_state)
    metrics_test = _eval.evaluate_denorm_metrics(model, test_loader, device, feature_mean, feature_std)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "feature_mean": float(feature_mean),
        "feature_std": float(feature_std),
        "val_loss_best": float(best_val),
        "metrics_test": metrics_test,
    }

    if trial_number is not None:
        checkpoint["trial_number"] = trial_number
    if trial_params is not None:
        checkpoint["trial_params"] = dict(trial_params)

    if checkpoint_path is None:
        if trial_number is None:
            checkpoint_path = "model_checkpoint.pt"
        else:
            checkpoint_path = _ckpt.trial_ckpt_path(cfg.checkpoints_dir, trial_number)

    _ckpt.save_checkpoint(checkpoint_path, checkpoint)

    return metrics_test, checkpoint_path, float(best_val)
