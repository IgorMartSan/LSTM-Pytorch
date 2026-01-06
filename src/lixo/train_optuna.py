"""
Projeto completo: LSTM para previsão de preços + Optuna multi-objective (MAE, RMSE, MAPE)

Otimiza hiperparâmetros com Optuna e retorna 3 métricas como objetivos (multi-objective):
- MAE
- RMSE
- MAPE

Extras:
- Split temporal train/val/test
- Normalização sem vazamento (mean/std apenas do bloco de treino)
- Early stopping por val_loss
- Optuna pruning por val_loss
- Salva fronteira de Pareto em CSV
- Salva um "best compromise" em best_pareto_lstm.pt
- Salva estudo em SQLite (optuna_study.db) para retomar

Observação:
- Esse pipeline usa apenas 1 feature (Close). É fácil expandir para OHLCV+indicadores depois.
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, Optional, List, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import yfinance as yf
import optuna
import pandas as pd


# =========================
# Dataset / Model
# =========================

class StockPriceDataset(Dataset):
    """Dataset com (janela, target)."""
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    """LSTM -> Linear (regressão do próximo valor)."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.regressor(last_hidden)


# =========================
# Config
# =========================

@dataclass
class TrainConfig:
    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    feature: str = "Close"

    # Split temporal
    train_ratio: float = 0.85            # bloco inicial para treino+val
    val_ratio_within_train: float = 0.15 # fatia final dentro do bloco de treino

    # Defaults (Optuna vai sobrescrever durante trials)
    sequence_length: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    # Treino
    max_epochs: int = 120
    patience: int = 20

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Outputs
    study_name: str = "stock_lstm_optuna"
    storage_path: str = "sqlite:///optuna_study.db"
    pareto_csv_path: str = "pareto_trials.csv"
    best_checkpoint_path: str = "best_pareto_lstm.pt"


# =========================
# Data
# =========================

def download_price_series(symbol: str, start_date: str, end_date: str, feature: str) -> torch.Tensor:
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' não encontrada. Colunas: {list(df.columns)}")
    values = torch.tensor(df[feature].values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("Dataset retornou vazio. Verifique símbolo e datas.")
    return values


def normalize_with_stats(series: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    safe_std = std if std > 1e-12 else 1.0
    return (series - mean) / safe_std


def compute_train_stats(series: torch.Tensor, train_ratio: float) -> Tuple[float, float]:
    train_size = max(int(series.size(0) * train_ratio), 1)
    train_slice = series[:train_size]
    mean = train_slice.mean().item()
    std = train_slice.std(unbiased=False).item()
    std = std if std > 1e-12 else 1.0
    return mean, std


def create_window_tensors(series: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converte série 1D em:
      inputs: (N, seq_len, 1)
      targets: (N, 1)
    """
    series = series.flatten()
    if series.size(0) <= sequence_length:
        raise ValueError("Série insuficiente para janelas. Reduza sequence_length.")
    inputs, targets = [], []
    for i in range(series.size(0) - sequence_length):
        window = series[i:i + sequence_length].unsqueeze(-1)
        target = series[i + sequence_length].unsqueeze(-1)
        inputs.append(window)
        targets.append(target)
    return torch.stack(inputs), torch.stack(targets)


def split_train_val_test(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float,
    val_ratio_within_train: float,
) -> Tuple[StockPriceDataset, StockPriceDataset, StockPriceDataset]:
    """
    Split temporal: [train][val][test]
    - train_ratio define o bloco inicial train+val
    - val_ratio_within_train define a fatia final do bloco train+val como validação
    """
    n = inputs.size(0)
    train_end = max(int(n * train_ratio), 1)
    if train_end >= n:
        raise ValueError("Sem exemplos reservados para teste. Ajuste train_ratio.")

    train_inputs_full = inputs[:train_end]
    train_targets_full = targets[:train_end]

    val_size = max(int(train_inputs_full.size(0) * val_ratio_within_train), 1)
    if train_inputs_full.size(0) - val_size < 1:
        raise ValueError("Treino muito pequeno para separar validação. Ajuste ratios/sequence_length.")

    train_inputs = train_inputs_full[:-val_size]
    train_targets = train_targets_full[:-val_size]
    val_inputs = train_inputs_full[-val_size:]
    val_targets = train_targets_full[-val_size:]

    test_inputs = inputs[train_end:]
    test_targets = targets[train_end:]
    if test_inputs.size(0) == 0:
        raise ValueError("Teste vazio. Ajuste train_ratio.")

    return (
        StockPriceDataset(train_inputs, train_targets),
        StockPriceDataset(val_inputs, val_targets),
        StockPriceDataset(test_inputs, test_targets),
    )


# =========================
# Metrics (denorm)
# =========================

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

    preds_denorm = preds * std + mean
    trues_denorm = trues * std + mean

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


# =========================
# Train single trial
# =========================

def train_single_trial(
    cfg: TrainConfig,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    feature_mean: float,
    feature_std: float,
    trial: optuna.Trial,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Treina com early stopping e pruning (por val_loss).
    Retorna:
      - metrics_test: MAE, RMSE, MAPE (escala real)
      - checkpoint: dict com model_state, cfg, stats, etc.
    """
    device = torch.device(cfg.device)

    train_ds, val_ds, test_ds = split_train_val_test(
        inputs, targets, cfg.train_ratio, cfg.val_ratio_within_train
    )

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

        train_loss = running / max(n, 1)
        val_loss = compute_val_loss_mse(model, val_loader, device)

        # ✅ sem trial.report (multi-objective não suporta)
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
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "val_loss_best": best_val,
        "trial_params": dict(trial.params),
        "metrics_test": metrics_test,
    }
    return metrics_test, checkpoint


# =========================
# Optuna multi-objective
# =========================

def choose_best_from_pareto(pareto_trials: List[optuna.trial.FrozenTrial], strategy: str = "min_mape") -> optuna.trial.FrozenTrial:
    """
    Define qual trial da fronteira de Pareto será salvo como "best".
    - "min_mape": escolhe menor MAPE
    - "min_rmse": escolhe menor RMSE
    - "min_mae": escolhe menor MAE
    - "weighted": soma ponderada (exemplo simples)
    """
    if not pareto_trials:
        raise ValueError("Pareto vazio.")

    if strategy == "min_mae":
        return min(pareto_trials, key=lambda t: t.values[0])
    if strategy == "min_rmse":
        return min(pareto_trials, key=lambda t: t.values[1])
    if strategy == "min_mape":
        return min(pareto_trials, key=lambda t: t.values[2])

    if strategy == "weighted":
        # pesos simples (ajuste ao seu gosto)
        w_mae, w_rmse, w_mape = 1.0, 1.0, 1.0
        return min(pareto_trials, key=lambda t: w_mae*t.values[0] + w_rmse*t.values[1] + w_mape*t.values[2])

    raise ValueError(f"Estratégia desconhecida: {strategy}")


def export_pareto_csv(study: optuna.Study, path: str) -> None:
    rows = []
    for t in study.best_trials:
        rows.append({
            "trial_number": t.number,
            "MAE": t.values[0],
            "RMSE": t.values[1],
            "MAPE": t.values[2],
            "val_loss_best": t.user_attrs.get("val_loss_best"),
            "params": t.params,
        })
    df = pd.DataFrame(rows).sort_values(by="MAPE", ascending=True)
    df.to_csv(path, index=False)


def run_optuna(cfg: TrainConfig, n_trials: int = 30, timeout_sec: Optional[int] = None, best_strategy: str = "min_mape") -> None:
    print("Baixando dados...")
    raw = download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)

    print("Normalização sem vazamento (stats só do bloco de treino)...")
    mean, std = compute_train_stats(raw, cfg.train_ratio)
    normalized = normalize_with_stats(raw, mean, std)

    # Storage para retomar
    storage = optuna.storages.RDBStorage(url=cfg.storage_path)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    # Multi-objective: (MAE, RMSE, MAPE) — todos minimizados
    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],
        # pruner=pruner,
    )
    study.set_metric_names(["MAE", "RMSE", "MAPE"])
    

    def objective(trial: optuna.Trial):
        # Espaço de busca (ajuste ao seu gosto)
        cfg_local = TrainConfig(**asdict(cfg))  # isola alterações por trial

        cfg_local.sequence_length = trial.suggest_int("sequence_length", 20, 120, step=10)
        cfg_local.hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        cfg_local.num_layers = trial.suggest_int("num_layers", 1, 4)
        cfg_local.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        cfg_local.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        cfg_local.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        cfg_local.max_epochs = trial.suggest_int("max_epochs", 30, 120, step=30)
        cfg_local.patience = trial.suggest_int("patience", 10, 25, step=5)

        # Depende do sequence_length
        inputs, targets = create_window_tensors(normalized, cfg_local.sequence_length)

        metrics_test, ckpt = train_single_trial(
            cfg=cfg_local,
            inputs=inputs,
            targets=targets,
            feature_mean=mean,
            feature_std=std,
            trial=trial,
        )

        # Guarda infos no trial
        trial.set_user_attr("val_loss_best", ckpt["val_loss_best"])
        trial.set_user_attr("metrics_test", metrics_test)
        trial.set_user_attr("metrics_test", metrics_test)   # dict de floats OK
        trial.set_user_attr("val_loss_best", float(ckpt["val_loss_best"]))

        # Retorna 3 objetivos: MAE, RMSE, MAPE
        return metrics_test["MAE"], metrics_test["RMSE"], metrics_test["MAPE"]

    print(f"Iniciando Optuna: trials={n_trials}, timeout={timeout_sec}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    # Export Pareto
    export_pareto_csv(study, cfg.pareto_csv_path)
    print(f"Pareto exportado em: {cfg.pareto_csv_path}")

    # Escolhe um trial da fronteira para salvar como "best"
    pareto = study.best_trials
    best_trial = choose_best_from_pareto(pareto, strategy=best_strategy)

    print("\n===== MELHOR TRIAL ESCOLHIDO (PARETO) =====")
    print(f"trial_number: {best_trial.number}")
    print(f"MAE : {best_trial.values[0]:.6f}")
    print(f"RMSE: {best_trial.values[1]:.6f}")
    print(f"MAPE: {best_trial.values[2]:.6f}")
    print("val_loss_best:", best_trial.user_attrs.get("val_loss_best"))
    print("\nParams:")
    for k, v in best_trial.params.items():
        print(f"  - {k}: {v}")

    ckpt = best_trial.user_attrs.get("checkpoint")
    if ckpt is None:
        print("\n[AVISO] Não achei checkpoint no best_trial.user_attrs. (pode ocorrer se trial antigo no DB não salvava attrs)")
        return

    torch.save(ckpt, cfg.best_checkpoint_path)
    print(f"\nCheckpoint salvo em: {cfg.best_checkpoint_path}")


# =========================
# Load for inference (exemplo)
# =========================

def load_checkpoint(path: str, device: Optional[str] = None):
    ckpt = torch.load(path, map_location="cpu")
    cfg_dict = ckpt["config"]
    mean = ckpt["feature_mean"]
    std = ckpt["feature_std"]

    cfg = TrainConfig(**cfg_dict)
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


# =========================
# Main
# =========================

if __name__ == "__main__":
    cfg = TrainConfig(
        symbol="DIS",
        start_date="2018-01-01",
        end_date="2024-07-20",
        feature="Close",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Rode:
    # - n_trials: número de tentativas
    # - best_strategy: como escolher 1 modelo da fronteira (min_mape/min_rmse/min_mae/weighted)
    run_optuna(cfg, n_trials=2000, timeout_sec=None, best_strategy="min_mape")

    # Exemplo de load:
    # model, cfg_loaded, mean, std, ckpt = load_checkpoint(cfg.best_checkpoint_path)
