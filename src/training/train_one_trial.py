from dataclasses import asdict
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import optuna

from config import TrainConfig
from models.lstm import LSTMRegressor
from training.evaluate import compute_val_loss_mse, evaluate_denorm_metrics
from training.checkpointing import save_checkpoint, trial_ckpt_path

def train_single_trial(
    cfg: TrainConfig,
    train_tensors,
    val_tensors,
    test_tensors,
    feature_mean: float,
    feature_std: float,
    trial: optuna.Trial,
) -> Tuple[Dict[str, float], str]:
    """
    Retorna:
      - metrics_test (real scale): dict MAE/RMSE/MAPE
      - ckpt_file: caminho do checkpoint salvo em disco
    """
    device = torch.device(cfg.device)

    train_ds = TensorDataset(train_tensors[0], train_tensors[1])
    val_ds = TensorDataset(val_tensors[0], val_tensors[1])
    test_ds = TensorDataset(test_tensors[0], test_tensors[1])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = LSTMRegressor(
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

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item() * x.size(0)
            n += x.size(0)

        val_loss = compute_val_loss_mse(model, val_loader, device)

        # (Opcional) pruning: Optuna precisa de trial.report + trial.should_prune
        # e isso FUNCIONA mesmo em multi-objective em versões recentes,
        # mas se você preferir evitar, deixe desativado.
        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            running = 0.0
            n = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running += loss.item() * x.size(0)
                n += x.size(0)

            val_loss = compute_val_loss_mse(model, val_loader, device)

            # ✅ Multi-objective: sem trial.report / pruning
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Falha: best_state não foi definido.")

    model.load_state_dict(best_state)

    metrics_test = evaluate_denorm_metrics(model, test_loader, device, feature_mean, feature_std)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "feature_mean": float(feature_mean),
        "feature_std": float(feature_std),
        "val_loss_best": float(best_val),
        "trial_params": dict(trial.params),
        "metrics_test": metrics_test,
        "trial_number": trial.number,
    }

    ckpt_file = trial_ckpt_path(cfg.checkpoints_dir, trial.number)
    save_checkpoint(ckpt_file, checkpoint)

    # guardar o caminho do checkpoint no trial para recuperar depois
    trial.set_user_attr("checkpoint_path", ckpt_file)
    trial.set_user_attr("val_loss_best", float(best_val))

    return metrics_test, ckpt_file
